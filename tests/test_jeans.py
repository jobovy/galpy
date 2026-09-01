# Tests of the galpy.df.jeans module: Jeans equations
import numpy

from galpy.df import jeans


# Test sigmar: radial velocity dispersion from the spherical Jeans equation
# For log halo, constant beta: sigma(r) = vc/sqrt(2.-2*beta)
def test_sigmar_wlog_constbeta():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    rs = numpy.linspace(0.001, 5.0, 101)
    # beta = 0 --> sigma = vc/sqrt(2)
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmar(lp, r) for r in rs]) - 1.0 / numpy.sqrt(2.0)
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    )
    # general beta --> sigma = vc/sqrt(2-2beta)
    beta = 0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmar(lp, r, beta=beta) for r in rs])
            - 1.0 / numpy.sqrt(2.0 - 2.0 * beta)
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0.5"
    )
    beta = -0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmar(lp, r, beta=beta) for r in rs])
            - 1.0 / numpy.sqrt(2.0 - 2.0 * beta)
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=-0.5"
    )
    return None


# Test sigmar: radial velocity dispersion from the spherical Jeans equation
# For log halo, constant beta: sigma(r) = vc/sqrt(2.-2*beta)
def test_sigmar_wlog_constbeta_diffdens_powerlaw():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    rs = numpy.linspace(0.001, 5.0, 101)
    # general beta and r^-gamma --> sigma = vc/sqrt(gamma-2beta)
    gamma, beta = 1.0, 0.0
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [jeans.sigmar(lp, r, beta=beta, dens=lambda r: r**-gamma) for r in rs]
            )
            - 1.0 / numpy.sqrt(gamma - 2.0 * beta)
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential, beta=0, and power-law density r^-1"
    )
    gamma, beta = 3.0, 0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [jeans.sigmar(lp, r, beta=beta, dens=lambda r: r**-gamma) for r in rs]
            )
            - 1.0 / numpy.sqrt(gamma - 2.0 * beta)
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential, beta=0.5, and power-law density r^-3"
    )
    gamma, beta = 0.0, -0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [jeans.sigmar(lp, r, beta=beta, dens=lambda r: r**-gamma) for r in rs]
            )
            - 1.0 / numpy.sqrt(gamma - 2.0 * beta)
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential, beta=-0.5, and power-law density r^0"
    )
    return None


def test_sigmar_wlog_constbeta_asbetafunc():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    rs = numpy.linspace(0.001, 5.0, 101)
    # beta = 0 --> sigma = vc/sqrt(2)
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmar(lp, r, beta=lambda x: 0.0) for r in rs])
            - 1.0 / numpy.sqrt(2.0)
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    )
    # general beta --> sigma = vc/sqrt(2-2beta)
    beta = lambda x: 0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmar(lp, r, beta=beta) for r in rs])
            - 1.0 / numpy.sqrt(2.0 - 2.0 * beta(0))
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0.5"
    )
    beta = lambda x: -0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmar(lp, r, beta=beta) for r in rs])
            - 1.0 / numpy.sqrt(2.0 - 2.0 * beta(0))
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=-0.5"
    )
    return None


def test_sigmar_wlog_linbeta():
    # for log halo, dens ~ r^-gamma, and beta = -b x r -->
    # sigmar = vc sqrt( scipy.special.gamma(-gamma)*scipy.special.gammaincc(-gamma,2*b*r)/[(2*b*r)**-gamma*exp(-2*b*r)]
    from scipy import special

    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    rs = numpy.linspace(0.001, 5.0, 101)
    gamma, b = -0.1, 3.0
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [
                    jeans.sigmar(lp, r, beta=lambda x: -b * x, dens=lambda x: x**-gamma)
                    - numpy.sqrt(
                        special.gamma(-gamma)
                        * special.gammaincc(-gamma, 2 * b * r)
                        / ((2 * b * r) ** -gamma * numpy.exp(-2.0 * b * r))
                    )
                    for r in rs
                ]
            )
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential, beta= -b*r, and dens ~ r^-gamma"
    )
    gamma, b = -0.5, 4.0
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [
                    jeans.sigmar(lp, r, beta=lambda x: -b * x, dens=lambda x: x**-gamma)
                    - numpy.sqrt(
                        special.gamma(-gamma)
                        * special.gammaincc(-gamma, 2 * b * r)
                        / ((2 * b * r) ** -gamma * numpy.exp(-2.0 * b * r))
                    )
                    for r in rs
                ]
            )
        )
        < 1e-10
    ), (
        "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential, beta= -b*r, and dens ~ r^-gamma"
    )
    return None


# Test sigmalos: radial velocity dispersion from the spherical Jeans equation
# For log halo, beta = 0: sigmalos(r) = vc/sqrt(2.)
def test_sigmalos_wlog_zerobeta():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    rs = numpy.linspace(0.5, 2.0, 3)
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmalos(lp, r) for r in rs]) - 1.0 / numpy.sqrt(2.0)
        )
        < 1e-8
    ), (
        "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    )
    # Also with pre-computed sigmar
    rs = numpy.linspace(0.5, 2.0, 11)
    # beta = 0 --> sigma = vc/sqrt(2)
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [jeans.sigmalos(lp, r, sigma_r=1.0 / numpy.sqrt(2.0)) for r in rs]
            )
            - 1.0 / numpy.sqrt(2.0)
        )
        < 1e-8
    ), (
        "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    )
    # Also with pre-computed, callable sigmar
    rs = numpy.linspace(0.5, 2.0, 11)
    # beta = 0 --> sigma = vc/sqrt(2)
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [
                    jeans.sigmalos(lp, r, sigma_r=lambda x: 1.0 / numpy.sqrt(2.0))
                    for r in rs
                ]
            )
            - 1.0 / numpy.sqrt(2.0)
        )
        < 1e-8
    ), (
        "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    )
    # Also with pre-computed, callable sigmar and dens given
    rs = numpy.linspace(0.5, 2.0, 11)
    # beta = 0 --> sigma = vc/sqrt(2)
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [
                    jeans.sigmalos(
                        lp,
                        r,
                        dens=lambda x: x**-2,
                        sigma_r=lambda x: 1.0 / numpy.sqrt(2.0),
                    )
                    for r in rs
                ]
            )
            - 1.0 / numpy.sqrt(2.0)
        )
        < 1e-8
    ), (
        "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    )
    # Also with pre-computed, callable sigmar and dens,surfdens given as func
    rs = numpy.linspace(0.5, 2.0, 11)
    # beta = 0 --> sigma = vc/sqrt(2)
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [
                    jeans.sigmalos(
                        lp,
                        r,
                        dens=lambda x: lp.dens(x, 0.0),
                        surfdens=lambda x: lp.surfdens(x, numpy.inf),
                        sigma_r=lambda x: 1.0 / numpy.sqrt(2.0),
                    )
                    for r in rs
                ]
            )
            - 1.0 / numpy.sqrt(2.0)
        )
        < 1e-8
    ), (
        "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    )
    # Also with pre-computed, callable sigmar and dens,surfdens given (value)
    rs = numpy.linspace(0.5, 2.0, 11)
    # beta = 0 --> sigma = vc/sqrt(2)
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [
                    jeans.sigmalos(
                        lp,
                        r,
                        dens=lambda x: lp.dens(x, 0.0),
                        surfdens=lp.surfdens(r, numpy.inf),
                        sigma_r=lambda x: 1.0 / numpy.sqrt(2.0),
                    )
                    for r in rs
                ]
            )
            - 1.0 / numpy.sqrt(2.0)
        )
        < 1e-8
    ), (
        "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    )
    # Also with pre-computed sigmar and callable beta
    rs = numpy.linspace(0.5, 2.0, 11)
    # beta = 0 --> sigma = vc/sqrt(2)
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [
                    jeans.sigmalos(
                        lp, r, sigma_r=1.0 / numpy.sqrt(2.0), beta=lambda x: 0.0
                    )
                    for r in rs
                ]
            )
            - 1.0 / numpy.sqrt(2.0)
        )
        < 1e-8
    ), (
        "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    )
    return None


def test_sigmar_on_grid_matches_the_per_radius_loop():
    # The cumulative fast path must agree with one-adaptive-quad-per-radius.
    # MWPotential2014 is REQUIRED here, not decorative: Hernquist and NFW are
    # insensitive to the large-r cancellation that the outside-in accumulation
    # avoids, so a test using only those would pass with that bug present
    # (measured: MWPotential2014 drifts ~5e-06 with it, ~7e-07 without).
    import numpy

    from galpy import potential
    from galpy.df import jeans

    for Pot in (
        potential.MWPotential2014,
        potential.NFWPotential(normalize=1.0, a=1.5),
    ):
        rs = numpy.linspace(1e-4, 25.0, 101)
        fast = jeans._sigmar_on_grid(Pot, rs, beta=0.0)
        assert fast is not None, "fast path unexpectedly declined"
        for ii in range(0, len(rs), 25):
            ref = jeans.sigmar(Pot, rs[ii], beta=0.0, use_physical=False)
            assert numpy.fabs(fast[ii] - ref) / numpy.fabs(ref) < 2e-6, (
                f"sigma_r disagrees with the per-radius quadrature at r={rs[ii]}"
            )


def test_sigmar_on_grid_matches_the_loop_for_an_expensive_integrand():
    # SCFPotential costs spherical harmonics at every point. It is the case an
    # approach needing a fine grid had to refuse (it was measured 3x SLOWER
    # there than the per-radius quadrature); interpolating the integrand puts it
    # back on the fast path, so check the ANSWER and not just that it accepted.
    import numpy

    from galpy import potential
    from galpy.df import jeans

    Pot = potential.SCFPotential(normalize=1.0)
    rs = numpy.linspace(1e-4, 25.0, 101)
    fast = jeans._sigmar_on_grid(Pot, rs, beta=0.0)
    assert fast is not None, "fast path unexpectedly declined for SCFPotential"
    for ii in range(0, len(rs), 25):
        ref = jeans.sigmar(Pot, rs[ii], beta=0.0, use_physical=False)
        assert numpy.fabs(fast[ii] - ref) / numpy.fabs(ref) < 2e-6, (
            f"sigma_r disagrees with the per-radius quadrature at r={rs[ii]}"
        )


def test_sigmar_on_grid_declines_when_it_cannot_converge():
    # The interpolated integrand is refined until the answer stops moving. A
    # DoubleExponentialDiskPotential has vertical structure along the r=(R,z)
    # ray that a log-spaced spline cannot resolve, so it never converges and
    # must decline rather than return a plausible-looking wrong answer. (It is
    # also not really an application of the SPHERICAL Jeans equation.)
    import numpy

    from galpy import potential
    from galpy.df import jeans

    rs = numpy.linspace(1e-4, 25.0, 101)
    assert (
        jeans._sigmar_on_grid(
            potential.DoubleExponentialDiskPotential(normalize=0.2, hr=3.0, hz=0.6), rs
        )
        is None
    ), "must decline when the interpolated integrand does not converge"
    # ...while an expensive-but-smooth integrand is accepted: SCFPotential costs
    # spherical harmonics per point, which ruled it out under an approach that
    # needed a fine grid, but it converges fine on ~1000 samples.
    assert (
        jeans._sigmar_on_grid(potential.SCFPotential(normalize=1.0), rs) is not None
    ), "an expensive but well-resolved integrand should still use the fast path"


def test_sigmar_on_grid_declines_outside_its_assumptions():
    # The grid is geometric and the tail is one quadrature from max(rs), so the
    # fast path only applies to an increasing, strictly positive radius range
    # with a constant anisotropy. Anything else must decline rather than
    # silently produce a grid it cannot represent.
    import numpy

    from galpy import potential
    from galpy.df import jeans

    hp = potential.HernquistPotential(normalize=1.0, a=2.0)
    assert jeans._sigmar_on_grid(hp, numpy.linspace(0.0, 25.0, 51)) is None  # r=0
    assert jeans._sigmar_on_grid(hp, numpy.linspace(5.0, 5.0, 51)) is None  # no range
    assert (
        jeans._sigmar_on_grid(hp, numpy.linspace(1e-4, 25.0, 51), beta=lambda r: 0.1)
        is None
    )  # callable beta
