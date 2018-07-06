# Tests of the galpy.df.jeans module: Jeans equations
import numpy
from galpy.df import jeans

# Test sigmar: radial velocity dispersion from the spherical Jeans equation
# For log halo, constant beta: sigma(r) = vc/sqrt(2.-2*beta)
def test_sigmar_wlog_constbeta():
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=1.)
    rs= numpy.linspace(0.001,5.,101)
    # beta = 0 --> sigma = vc/sqrt(2)
    assert numpy.all(numpy.fabs(numpy.array([jeans.sigmar(lp,r) for r in rs])-1./numpy.sqrt(2.)) < 1e-10), 'Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0'
    # general beta --> sigma = vc/sqrt(2-2beta)
    beta= 0.5
    assert numpy.all(numpy.fabs(numpy.array([jeans.sigmar(lp,r,beta=beta) for r in rs])-1./numpy.sqrt(2.-2.*beta)) < 1e-10), 'Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0'
    beta= -0.5
    assert numpy.all(numpy.fabs(numpy.array([jeans.sigmar(lp,r,beta=beta) for r in rs])-1./numpy.sqrt(2.-2.*beta)) < 1e-10), 'Radial sigma computed w/ spherical Jeans equation incorrect for LogarithmicHaloPotential and beta=0'
    return None
