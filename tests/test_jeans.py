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
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    # general beta --> sigma = vc/sqrt(2-2beta)
    beta = 0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmar(lp, r, beta=beta) for r in rs])
            - 1.0 / numpy.sqrt(2.0 - 2.0 * beta)
        )
        < 1e-10
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0.5"
    beta = -0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmar(lp, r, beta=beta) for r in rs])
            - 1.0 / numpy.sqrt(2.0 - 2.0 * beta)
        )
        < 1e-10
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=-0.5"
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
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential, beta=0, and power-law density r^-1"
    gamma, beta = 3.0, 0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [jeans.sigmar(lp, r, beta=beta, dens=lambda r: r**-gamma) for r in rs]
            )
            - 1.0 / numpy.sqrt(gamma - 2.0 * beta)
        )
        < 1e-10
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential, beta=0.5, and power-law density r^-3"
    gamma, beta = 0.0, -0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [jeans.sigmar(lp, r, beta=beta, dens=lambda r: r**-gamma) for r in rs]
            )
            - 1.0 / numpy.sqrt(gamma - 2.0 * beta)
        )
        < 1e-10
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential, beta=-0.5, and power-law density r^0"
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
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    # general beta --> sigma = vc/sqrt(2-2beta)
    beta = lambda x: 0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmar(lp, r, beta=beta) for r in rs])
            - 1.0 / numpy.sqrt(2.0 - 2.0 * beta(0))
        )
        < 1e-10
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0.5"
    beta = lambda x: -0.5
    assert numpy.all(
        numpy.fabs(
            numpy.array([jeans.sigmar(lp, r, beta=beta) for r in rs])
            - 1.0 / numpy.sqrt(2.0 - 2.0 * beta(0))
        )
        < 1e-10
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=-0.5"
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
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential, beta= -b*r, and dens ~ r^-gamma"
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
    ), "Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential, beta= -b*r, and dens ~ r^-gamma"
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
    ), "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
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
    ), "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
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
    ), "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
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
    ), "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
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
    ), "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
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
    ), "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
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
    ), "Radial sigma_los computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0"
    return None
