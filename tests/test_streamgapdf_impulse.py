# Tests of the various variations on the impulse approx. from Sanders, Bovy,
# & Erkal (2016)
import numpy
import pytest


# Test the Plummer calculation for a perpendicular impact, B&T ex. 8.7
def test_impulse_deltav_plummer_subhalo_perpendicular():
    from galpy.df import impulse_deltav_plummer

    tol = -10.0
    kick = impulse_deltav_plummer(
        numpy.array([[0.0, numpy.pi, 0.0]]),
        numpy.array([0.0]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        1.5,
        4.0,
    )
    # Should be B&T (8.152)
    assert (
        numpy.fabs(kick[0, 0] - 2.0 * 1.5 * 3.0 / numpy.pi * 2.0 / 25.0) < 10.0**tol
    ), "Perpendicular kick of subhalo perpendicular not as expected"
    assert (
        numpy.fabs(kick[0, 2] + 2.0 * 1.5 * 3.0 / numpy.pi * 2.0 / 25.0) < 10.0**tol
    ), "Perpendicular kick of subhalo perpendicular not as expected"
    # Same for along z
    kick = impulse_deltav_plummer(
        numpy.array([[0.0, 0.0, numpy.pi]]),
        numpy.array([0.0]),
        3.0,
        numpy.array([0.0, 0.0, numpy.pi / 2.0]),
        1.5,
        4.0,
    )
    # Should be B&T (8.152)
    assert (
        numpy.fabs(kick[0, 0] - 2.0 * 1.5 * 3.0 / numpy.pi * 2.0 / 25.0) < 10.0**tol
    ), "Perpendicular kick of subhalo perpendicular not as expected"
    assert (
        numpy.fabs(kick[0, 1] - 2.0 * 1.5 * 3.0 / numpy.pi * 2.0 / 25.0) < 10.0**tol
    ), "Perpendicular kick of subhalo perpendicular not as expected"
    return None


# Test the Plummer curved calculation for a perpendicular impact
def test_impulse_deltav_plummer_curved_subhalo_perpendicular():
    from galpy.df import impulse_deltav_plummer, impulse_deltav_plummer_curvedstream

    tol = -10.0
    kick = impulse_deltav_plummer(
        numpy.array([[3.4, 0.0, 0.0]]),
        numpy.array([4.0]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        1.5,
        4.0,
    )
    curved_kick = impulse_deltav_plummer_curvedstream(
        numpy.array([[3.4, 0.0, 0.0]]),
        numpy.array([[4.0, 0.0, 0.0]]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([3.4, 0.0, 0.0]),
        1.5,
        4.0,
    )
    # Should be equal
    assert numpy.all(
        numpy.fabs(kick - curved_kick) < 10.0**tol
    ), "curved Plummer kick does not agree with straight kick for straight track"
    # Same for a bunch of positions
    v = numpy.zeros((100, 3))
    v[:, 0] = 3.4
    xpos = numpy.random.normal(size=100)
    kick = impulse_deltav_plummer(
        v, xpos, 3.0, numpy.array([0.0, numpy.pi / 2.0, 0.0]), 1.5, 4.0
    )
    xpos = numpy.array([xpos, numpy.zeros(100), numpy.zeros(100)]).T
    curved_kick = impulse_deltav_plummer_curvedstream(
        v,
        xpos,
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([3.4, 0.0, 0.0]),
        1.5,
        4.0,
    )
    # Should be equal
    assert numpy.all(
        numpy.fabs(kick - curved_kick) < 10.0**tol
    ), "curved Plummer kick does not agree with straight kick for straight track"
    return None


# Test general impulse vs. Plummer
def test_impulse_deltav_general():
    from galpy.df import impulse_deltav_general, impulse_deltav_plummer
    from galpy.potential import PlummerPotential

    tol = -10.0
    kick = impulse_deltav_plummer(
        numpy.array([[3.4, 0.0, 0.0]]),
        numpy.array([4.0]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        1.5,
        4.0,
    )
    pp = PlummerPotential(amp=1.5, b=4.0)
    general_kick = impulse_deltav_general(
        numpy.array([[3.4, 0.0, 0.0]]),
        numpy.array([4.0]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        pp,
    )
    assert numpy.all(
        numpy.fabs(kick - general_kick) < 10.0**tol
    ), "general kick calculation does not agree with Plummer calculation for a Plummer potential"
    # Same for a bunch of positions
    v = numpy.zeros((100, 3))
    v[:, 0] = 3.4
    xpos = numpy.random.normal(size=100)
    kick = impulse_deltav_plummer(
        v, xpos, 3.0, numpy.array([0.0, numpy.pi / 2.0, 0.0]), numpy.pi, numpy.exp(1.0)
    )
    pp = PlummerPotential(amp=numpy.pi, b=numpy.exp(1.0))
    general_kick = impulse_deltav_general(
        v, xpos, 3.0, numpy.array([0.0, numpy.pi / 2.0, 0.0]), pp
    )
    assert numpy.all(
        numpy.fabs(kick - general_kick) < 10.0**tol
    ), "general kick calculation does not agree with Plummer calculation for a Plummer potential"
    return None


# Test general impulse vs. Plummer for curved stream
def test_impulse_deltav_general_curved():
    from galpy.df import (
        impulse_deltav_general_curvedstream,
        impulse_deltav_plummer_curvedstream,
    )
    from galpy.potential import PlummerPotential

    tol = -10.0
    kick = impulse_deltav_plummer_curvedstream(
        numpy.array([[3.4, 0.0, 0.0]]),
        numpy.array([[4.0, 0.0, 0.0]]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([3.4, 0.0, 0.0]),
        1.5,
        4.0,
    )
    pp = PlummerPotential(amp=1.5, b=4.0)
    general_kick = impulse_deltav_general_curvedstream(
        numpy.array([[3.4, 0.0, 0.0]]),
        numpy.array([[4.0, 0.0, 0.0]]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([3.4, 0.0, 0.0]),
        pp,
    )
    assert numpy.all(
        numpy.fabs(kick - general_kick) < 10.0**tol
    ), "general kick calculation does not agree with Plummer calculation for a Plummer potential, for curved stream"
    # Same for a bunch of positions
    v = numpy.zeros((100, 3))
    v[:, 0] = 3.4
    xpos = numpy.random.normal(size=100)
    xpos = numpy.array([xpos, numpy.zeros(100), numpy.zeros(100)]).T
    kick = impulse_deltav_plummer_curvedstream(
        v,
        xpos,
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([3.4, 0.0, 0.0]),
        numpy.pi,
        numpy.exp(1.0),
    )
    pp = PlummerPotential(amp=numpy.pi, b=numpy.exp(1.0))
    general_kick = impulse_deltav_general_curvedstream(
        v,
        xpos,
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([3.4, 0.0, 0.0]),
        pp,
    )
    assert numpy.all(
        numpy.fabs(kick - general_kick) < 10.0**tol
    ), "general kick calculation does not agree with Plummer calculation for a Plummer potential, for curved stream"
    return None


# Test general impulse vs. Hernquist
def test_impulse_deltav_general_hernquist():
    from galpy.df import impulse_deltav_general, impulse_deltav_hernquist
    from galpy.potential import HernquistPotential

    GM = 1.5
    tol = -10.0
    kick = impulse_deltav_hernquist(
        numpy.array([3.4, 0.0, 0.0]),
        numpy.array([4.0]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        GM,
        4.0,
    )
    # Note factor of 2 in definition of GM and amp
    pp = HernquistPotential(amp=2.0 * GM, a=4.0)
    general_kick = impulse_deltav_general(
        numpy.array([3.4, 0.0, 0.0]),
        numpy.array([4.0]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        pp,
    )
    assert numpy.all(
        numpy.fabs(kick - general_kick) < 10.0**tol
    ), "general kick calculation does not agree with Hernquist calculation for a Hernquist potential"
    # Same for a bunch of positions
    GM = numpy.pi
    v = numpy.zeros((100, 3))
    v[:, 0] = 3.4
    xpos = numpy.random.normal(size=100)
    kick = impulse_deltav_hernquist(
        v, xpos, 3.0, numpy.array([0.0, numpy.pi / 2.0, 0.0]), GM, numpy.exp(1.0)
    )
    pp = HernquistPotential(amp=2.0 * GM, a=numpy.exp(1.0))
    general_kick = impulse_deltav_general(
        v, xpos, 3.0, numpy.array([0.0, numpy.pi / 2.0, 0.0]), pp
    )
    assert numpy.all(
        numpy.fabs(kick - general_kick) < 10.0**tol
    ), "general kick calculation does not agree with Hernquist calculation for a Hernquist potential"
    return None


# Test general impulse vs. Hernquist for curved stream
def test_impulse_deltav_general_curved_hernquist():
    from galpy.df import (
        impulse_deltav_general_curvedstream,
        impulse_deltav_hernquist_curvedstream,
    )
    from galpy.potential import HernquistPotential

    GM = 1.5
    tol = -10.0
    kick = impulse_deltav_hernquist_curvedstream(
        numpy.array([3.4, 0.0, 0.0]),
        numpy.array([4.0, 0.0, 0.0]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([3.4, 0.0, 0.0]),
        GM,
        4.0,
    )
    # Note factor of 2 in definition of GM and amp
    pp = HernquistPotential(amp=2.0 * GM, a=4.0)
    general_kick = impulse_deltav_general_curvedstream(
        numpy.array([3.4, 0.0, 0.0]),
        numpy.array([4.0, 0.0, 0.0]),
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([3.4, 0.0, 0.0]),
        pp,
    )
    assert numpy.all(
        numpy.fabs(kick - general_kick) < 10.0**tol
    ), "general kick calculation does not agree with Hernquist calculation for a Hernquist potential, for curved stream"
    # Same for a bunch of positions
    GM = numpy.pi
    v = numpy.zeros((100, 3))
    v[:, 0] = 3.4
    xpos = numpy.random.normal(size=100)
    xpos = numpy.array([xpos, numpy.zeros(100), numpy.zeros(100)]).T
    kick = impulse_deltav_hernquist_curvedstream(
        v,
        xpos,
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([3.4, 0.0, 0.0]),
        GM,
        numpy.exp(1.0),
    )
    pp = HernquistPotential(amp=2.0 * GM, a=numpy.exp(1.0))
    general_kick = impulse_deltav_general_curvedstream(
        v,
        xpos,
        3.0,
        numpy.array([0.0, numpy.pi / 2.0, 0.0]),
        numpy.array([0.0, 0.0, 0.0]),
        numpy.array([3.4, 0.0, 0.0]),
        pp,
    )
    assert numpy.all(
        numpy.fabs(kick - general_kick) < 10.0**tol
    ), "general kick calculation does not agree with Hernquist calculation for a Hernquist potential, for curved stream"
    return None


def test_hernquistX_negative():
    from galpy.df.streamgapdf import HernquistX

    with pytest.raises(ValueError) as excinfo:
        HernquistX(-1.0)
    return None


def test_hernquistX_unity():
    from galpy.df.streamgapdf import HernquistX

    assert (
        HernquistX(1.0) == 1.0
    ), "Hernquist X function not returning 1 with argument 1"
    return None


# Test general impulse vs. full orbit integration for zero force
def test_impulse_deltav_general_orbit_zeroforce():
    from galpy.df import (
        impulse_deltav_general_orbitintegration,
        impulse_deltav_plummer_curvedstream,
    )
    from galpy.potential import PlummerPotential

    tol = -6.0
    rcurv = 10.0
    vp = 220.0
    x0 = numpy.array([rcurv, 0.0, 0.0])
    v0 = numpy.array([0.0, vp, 0.0])
    w = numpy.array([1.0, numpy.pi / 2.0, 0.0])
    plummer_kick = impulse_deltav_plummer_curvedstream(v0, x0, 3.0, w, x0, v0, 1.5, 4.0)
    pp = PlummerPotential(amp=1.5, b=4.0)
    vang = vp / rcurv
    angrange = numpy.pi
    maxt = 5.0 * angrange / vang
    galpot = constantPotential()
    orbit_kick = impulse_deltav_general_orbitintegration(
        v0, x0, 3.0, w, x0, v0, pp, maxt, galpot
    )
    assert numpy.all(
        numpy.fabs(orbit_kick - plummer_kick) < 10.0**tol
    ), "general kick with acceleration calculation does not agree with Plummer calculation for a Plummer potential, for straight"
    # Same for a bunch of positions
    tol = -5.0
    pp = PlummerPotential(amp=numpy.pi, b=numpy.exp(1.0))
    theta = numpy.linspace(-numpy.pi / 4.0, numpy.pi / 4.0, 100)
    xc, yc = rcurv * numpy.cos(theta), rcurv * numpy.sin(theta)
    Xc = numpy.zeros((100, 3))
    Xc[:, 0] = xc
    Xc[:, 1] = yc
    vx, vy = -vp * numpy.sin(theta), vp * numpy.cos(theta)
    V = numpy.zeros((100, 3))
    V[:, 0] = vx
    V[:, 1] = vy
    plummer_kick = impulse_deltav_plummer_curvedstream(
        V, Xc, 3.0, w, x0, v0, numpy.pi, numpy.exp(1.0)
    )
    orbit_kick = impulse_deltav_general_orbitintegration(
        V, Xc, 3.0, w, x0, v0, pp, maxt, galpot
    )
    assert numpy.all(
        numpy.fabs(orbit_kick - plummer_kick) < 10.0**tol
    ), "general kick calculation does not agree with Plummer calculation for a Plummer potential, for curved stream"
    return None


# Test general impulse vs. full stream and halo integration for zero force
def test_impulse_deltav_general_fullintegration_zeroforce():
    from galpy.df import (
        impulse_deltav_general_fullplummerintegration,
        impulse_deltav_plummer_curvedstream,
    )

    tol = -3.0
    rcurv = 10.0
    vp = 220.0
    GM = 1.5
    rs = 4.0
    x0 = numpy.array([rcurv, 0.0, 0.0])
    v0 = numpy.array([0.0, vp, 0.0])
    w = numpy.array([1.0, numpy.pi / 4.0 * vp, 0.0])
    plummer_kick = impulse_deltav_plummer_curvedstream(v0, x0, 3.0, w, x0, v0, GM, rs)
    galpot = constantPotential()
    orbit_kick = impulse_deltav_general_fullplummerintegration(
        v0, x0, 3.0, w, x0, v0, galpot, GM, rs, tmaxfac=100.0, N=1000
    )
    nzeroIndx = numpy.fabs(plummer_kick) > 10.0**tol
    assert numpy.all(
        numpy.fabs((orbit_kick - plummer_kick) / plummer_kick)[nzeroIndx] < 10.0**tol
    ), "general kick with acceleration calculation does not agree with Plummer calculation for a Plummer potential, for straight"
    assert numpy.all(
        numpy.fabs(orbit_kick - plummer_kick)[True ^ nzeroIndx] < 10.0**tol
    ), "general kick with acceleration calculation does not agree with Plummer calculation for a Plummer potential, for straight"
    # Same for a bunch of positions
    tol = -2.5
    GM = numpy.pi
    rs = numpy.exp(1.0)
    theta = numpy.linspace(-numpy.pi / 4.0, numpy.pi / 4.0, 10)
    xc, yc = rcurv * numpy.cos(theta), rcurv * numpy.sin(theta)
    Xc = numpy.zeros((10, 3))
    Xc[:, 0] = xc
    Xc[:, 1] = yc
    vx, vy = -vp * numpy.sin(theta), vp * numpy.cos(theta)
    V = numpy.zeros((10, 3))
    V[:, 0] = vx
    V[:, 1] = vy
    plummer_kick = impulse_deltav_plummer_curvedstream(V, Xc, 3.0, w, x0, v0, GM, rs)
    orbit_kick = impulse_deltav_general_fullplummerintegration(
        V, Xc, 3.0, w, x0, v0, galpot, GM, rs, tmaxfac=100.0
    )
    nzeroIndx = numpy.fabs(plummer_kick) > 10.0**tol
    assert numpy.all(
        numpy.fabs((orbit_kick - plummer_kick) / plummer_kick)[nzeroIndx] < 10.0**tol
    ), "full stream+halo integration calculation does not agree with Plummer calculation for a Plummer potential, for curved stream"
    assert numpy.all(
        numpy.fabs(orbit_kick - plummer_kick)[True ^ nzeroIndx] < 10.0**tol
    ), "full stream+halo integration calculation does not agree with Plummer calculation for a Plummer potential, for curved stream"
    return None


# Test general impulse vs. full stream and halo integration for fast encounter
def test_impulse_deltav_general_fullintegration_fastencounter():
    from galpy.df import (
        impulse_deltav_general_fullplummerintegration,
        impulse_deltav_general_orbitintegration,
    )
    from galpy.potential import LogarithmicHaloPotential, PlummerPotential

    tol = -2.0
    GM = 1.5
    rs = 4.0
    x0 = numpy.array([1.5, 0.0, 0.0])
    v0 = numpy.array([0.0, 1.0, 0.0])  # circular orbit
    w = numpy.array([0.0, 0.0, 100.0])  # very fast compared to v=1
    lp = LogarithmicHaloPotential(normalize=1.0)
    pp = PlummerPotential(amp=GM, b=rs)
    orbit_kick = impulse_deltav_general_orbitintegration(
        v0, x0, 3.0, w, x0, v0, pp, 5.0 * numpy.pi, lp
    )
    full_kick = impulse_deltav_general_fullplummerintegration(
        v0, x0, 3.0, w, x0, v0, lp, GM, rs, tmaxfac=10.0, N=1000
    )
    # Kick should be in the X direction
    assert (
        numpy.fabs((orbit_kick - full_kick) / full_kick)[0, 0] < 10.0**tol
    ), "Acceleration kick does not agree with full-orbit-integration kick for fast encounter"
    assert numpy.all(
        numpy.fabs(orbit_kick - full_kick)[0, 1:] < 10.0**tol
    ), "Acceleration kick does not agree with full-orbit-integration kick for fast encounter"
    return None


# Test straight, stream impulse vs. Plummer, similar setup as Fig. 1 in
# stream paper
def test_impulse_deltav_plummerstream():
    from galpy.df import impulse_deltav_plummer, impulse_deltav_plummerstream
    from galpy.util import conversion

    V0, R0 = 220.0, 8.0
    GM = 10.0**-2.0 / conversion.mass_in_1010msol(V0, R0)
    rs = 0.625 / R0
    b = rs
    stream_phi = numpy.linspace(-numpy.pi / 2.0, numpy.pi / 2.0, 201)
    stream_r = 10.0 / R0
    stream_v = 220.0 / V0
    x_gc = stream_r * stream_phi
    v_gc = numpy.tile([0.000001, stream_v, 0.000001], (201, 1))
    w = numpy.array([0.0, 132.0, 176]) / V0
    wmag = numpy.sqrt(numpy.sum(w**2.0))
    tol = -5.0
    # Plummer sphere kick
    kick = impulse_deltav_plummer(v_gc[101], x_gc[101], -b, w, GM, rs)
    # Kick from stream with length 0.01 r_s (should be ~Plummer sphere)
    dt = 0.01 * rs * R0 / wmag / V0 * conversion.freq_in_kmskpc(V0, R0)
    stream_kick = impulse_deltav_plummerstream(
        v_gc[101], x_gc[101], -b, w, lambda t: GM / dt, rs, -dt / 2.0, dt / 2.0
    )
    assert numpy.all(numpy.fabs((kick - stream_kick) / kick) < 10.0**tol), (
        "Short stream impulse kick calculation does not agree with Plummer calculation by %g"
        % (numpy.amax(numpy.fabs((kick - stream_kick) / kick)))
    )
    # Same for a bunch of positions
    kick = impulse_deltav_plummer(v_gc, x_gc, -b, w, GM, rs)
    # Kick from stream with length 0.01 r_s (should be ~Plummer sphere)
    dt = 0.01 * rs * R0 / wmag / V0 * conversion.freq_in_kmskpc(V0, R0)
    stream_kick = impulse_deltav_plummerstream(
        v_gc, x_gc, -b, w, lambda t: GM / dt, rs, -dt / 2.0, dt / 2.0
    )
    assert numpy.all(
        (numpy.fabs((kick - stream_kick) / kick) < 10.0**tol)
        * (numpy.fabs(kick) >= 10**-4.0)
        + (numpy.fabs(kick - stream_kick) < 10**tol) * (numpy.fabs(kick) < 10**-4.0)
    ), f"Short stream impulse kick calculation does not agree with Plummer calculation by rel: {numpy.amax(numpy.fabs((kick-stream_kick)/kick)[numpy.fabs(kick) >= 10**-4.]):g}, abs: {numpy.amax(numpy.fabs(kick-stream_kick)[numpy.fabs(kick) < 10**-3.]):g}"


def test_impulse_deltav_plummerstream_tmaxerror():
    from galpy.df import impulse_deltav_plummer, impulse_deltav_plummerstream
    from galpy.util import conversion

    V0, R0 = 220.0, 8.0
    GM = 10.0**-2.0 / conversion.mass_in_1010msol(V0, R0)
    rs = 0.625 / R0
    b = rs
    stream_phi = numpy.linspace(-numpy.pi / 2.0, numpy.pi / 2.0, 201)
    stream_r = 10.0 / R0
    stream_v = 220.0 / V0
    x_gc = stream_r * stream_phi
    v_gc = numpy.tile([0.000001, stream_v, 0.000001], (201, 1))
    w = numpy.array([0.0, 132.0, 176]) / V0
    wmag = numpy.sqrt(numpy.sum(w**2.0))
    tol = -5.0
    # Same for infinite integration limits
    kick = impulse_deltav_plummer(v_gc[101], x_gc[101], -b, w, GM, rs)
    # Kick from stream with length 0.01 r_s (should be ~Plummer sphere)
    dt = 0.01 * rs * R0 / wmag / V0 * conversion.freq_in_kmskpc(V0, R0)
    with pytest.raises(ValueError) as excinfo:
        stream_kick = impulse_deltav_plummerstream(
            v_gc[101], x_gc[101], -b, w, lambda t: GM / dt, rs
        )
    return None


# Test the Plummer curved calculation for a perpendicular stream impact:
# short impact should be the same as a Plummer-sphere impact
def test_impulse_deltav_plummerstream_curved_subhalo_perpendicular():
    from galpy.df import (
        impulse_deltav_plummer_curvedstream,
        impulse_deltav_plummerstream_curvedstream,
    )
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion

    R0, V0 = 8.0, 220.0
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    tol = -5.0
    GM = 10.0**-2.0 / conversion.mass_in_1010msol(V0, R0)
    rs = 0.625 / R0
    dt = 0.01 * rs / (numpy.pi / 4.0)
    kick = impulse_deltav_plummer_curvedstream(
        numpy.array([0.5, 0.1, 0.2]),
        numpy.array([1.2, 0.0, 0.0]),
        rs,
        numpy.array([0.1, numpy.pi / 4.0, 0.1]),
        numpy.array([1.2, 0.0, 0.0]),
        numpy.array([0.5, 0.1, 0.2]),
        GM,
        rs,
    )
    stream_kick = impulse_deltav_plummerstream_curvedstream(
        numpy.array([[0.5, 0.1, 0.2]]),
        numpy.array([[1.2, 0.0, 0.0]]),
        numpy.array([0.0]),
        rs,
        numpy.array([0.1, numpy.pi / 4.0, 0.1]),
        numpy.array([1.2, 0.0, 0.0]),
        numpy.array([0.5, 0.1, 0.2]),
        lambda t: GM / dt,
        rs,
        lp,
        -dt / 2.0,
        dt / 2.0,
    )
    # Should be equal
    assert numpy.all(numpy.fabs((kick - stream_kick) / kick) < 10.0**tol), (
        "Curved, short Plummer-stream kick does not agree with curved Plummer-sphere kick by %g"
        % (numpy.amax(numpy.fabs((kick - stream_kick) / kick)))
    )
    # Also test with other array shape input for x and v
    kick = impulse_deltav_plummer_curvedstream(
        numpy.array([[0.5, 0.1, 0.2]]),
        numpy.array([[1.2, 0.0, 0.0]]),
        rs,
        numpy.array([0.1, numpy.pi / 4.0, 0.1]),
        numpy.array([1.2, 0.0, 0.0]),
        numpy.array([0.5, 0.1, 0.2]),
        GM,
        rs,
    )
    stream_kick = impulse_deltav_plummerstream_curvedstream(
        numpy.array([0.5, 0.1, 0.2]),
        numpy.array([1.2, 0.0, 0.0]),
        numpy.array([0.0]),
        rs,
        numpy.array([0.1, numpy.pi / 4.0, 0.1]),
        numpy.array([1.2, 0.0, 0.0]),
        numpy.array([0.5, 0.1, 0.2]),
        lambda t: GM / dt,
        rs,
        lp,
        -dt / 2.0,
        dt / 2.0,
    )
    assert numpy.all(numpy.fabs((kick - stream_kick) / kick) < 10.0**tol), (
        "Curved, short Plummer-stream kick does not agree with curved Plummer-sphere kick by %g"
        % (numpy.amax(numpy.fabs((kick - stream_kick) / kick)))
    )
    return None


from galpy.potential import Potential


class constantPotential(Potential):
    def __init__(self):
        Potential.__init__(self, amp=1.0)
        self.hasC = False
        return None

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        return 0.0

    def _zforce(self, R, z, phi=0.0, t=0.0):
        return 0.0
