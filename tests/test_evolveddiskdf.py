# Tests of the evolveddiskdf module
import numpy

from galpy.df import dehnendf, evolveddiskdf
from galpy.potential import (
    EllipticalDiskPotential,
    LogarithmicHaloPotential,
    SteadyLogSpiralPotential,
)

_GRIDPOINTS = 31
# globals to save the results from previous calculations to be re-used, pre-setting them allows one to skip tests
_maxi_surfacemass = 0.0672746475968
_maxi_meanvr = -0.000517132979969
_maxi_meanvt = 0.913328340109
_maxi_sigmar2 = 0.0457686414529
_maxi_sigmat2 = 0.0268245643697
_maxi_sigmart = -0.000541204894097


def test_mildnonaxi_meanvr_grid():
    # Test that for a close to axisymmetric potential, the mean vr is close to zero
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvr, grid = edf.meanvR(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvr) < 0.001
    ), "meanvR of evolveddiskdf for axisymmetric potential is not equal to zero"
    mvr = edf.meanvR(0.9, phi=0.2, integrate_method="rk6_c", grid=grid)
    assert (
        numpy.fabs(mvr) < 0.001
    ), "meanvR of evolveddiskdf for axisymmetric potential is not equal to zero when calculated with pre-computed grid"
    # Pre-compute surfmass and use it, first test that grid is properly returned when given
    smass, ngrid = edf.vmomentsurfacemass(
        0.9,
        0,
        0,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        gridpoints=_GRIDPOINTS,
        returnGrid=True,
    )
    assert (
        ngrid == grid
    ), "grid returned by vmomentsurfacemass w/ grid input is not the same as the input"
    # Pre-compute surfmass and use it
    nsmass = edf.vmomentsurfacemass(
        0.9,
        0,
        0,
        phi=0.2 * 180.0 / numpy.pi,
        integrate_method="rk6_c",
        grid=True,
        gridpoints=_GRIDPOINTS,
        deg=True,
    )
    assert (
        numpy.fabs(smass - nsmass) < 0.001
    ), "surfacemass computed w/ and w/o returnGrid are not the same"
    mvr = edf.meanvR(
        0.9, phi=0.2, integrate_method="rk6_c", grid=grid, surfacemass=smass
    )
    assert (
        numpy.fabs(mvr) < 0.001
    ), "meanvR of evolveddiskdf for axisymmetric potential is not equal to zero when calculated with pre-computed grid and surfacemass"
    global _maxi_meanvr
    _maxi_meanvr = mvr
    global _maxi_surfacemass
    _maxi_surfacemass = smass
    return None


def test_mildnonaxi_meanvr_direct():
    # Test that for an axisymmetric potential, the mean vr is close to zero
    # We do this for an axisymmetric potential, bc otherwise it takes too long
    idf = dehnendf(beta=0.0)
    pot = [LogarithmicHaloPotential(normalize=1.0)]
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvr = edf.meanvR(0.9, phi=0.2, integrate_method="rk6_c", grid=False)
    assert (
        numpy.fabs(mvr) < 0.001
    ), "meanvR of evolveddiskdf for axisymmetric potential is not equal to zero when calculated directly"
    return None


def test_mildnonaxi_meanvr_grid_tlist():
    # Test that for a close to axisymmetric potential, the mean vr is close to zero
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvr, grid = edf.meanvR(
        0.9,
        t=[0.0, -2.5, -5.0, -7.5, -10.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert numpy.all(
        numpy.fabs(mvr) < 0.003
    ), "meanvR of evolveddiskdf for axisymmetric potential is not equal to zero for list of times"
    mvr = edf.meanvR(
        0.9,
        t=[0.0, -2.5, -5.0, -7.5, -10.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
    )
    assert numpy.all(
        numpy.fabs(mvr) < 0.003
    ), "meanvR of evolveddiskdf for axisymmetric potential is not equal to zero when calculated with pre-computed grid for list of times"
    return None


def test_mildnonaxi_meanvt_grid():
    # Test that for a close to axisymmetric potential, the mean vt is close to that of the initial DF
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvt, grid = edf.meanvT(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf"
    mvt = edf.meanvT(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when calculated with pre-computed grid"
    global _maxi_meanvt
    _maxi_meanvt = mvt
    return None


def test_mildnonaxi_meanvt_hierarchgrid():
    # Test that for a close to axisymmetric potential, the mean vt is close to that of the initial DF
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvt, grid = edf.meanvT(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        hierarchgrid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when using hierarchgrid"
    mvt = edf.meanvT(
        0.9, phi=0.2, integrate_method="rk6_c", grid=grid, gridpoints=_GRIDPOINTS
    )
    assert (
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when calculated with pre-computed grid when using hierarchgrid"
    # Also test that the hierarchgrid is properly returned
    smass, ngrid = edf.vmomentsurfacemass(
        0.9,
        0,
        0,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        gridpoints=_GRIDPOINTS,
        returnGrid=True,
    )
    assert (
        ngrid == grid
    ), "hierarchical grid returned by vmomentsurfacemass w/ grid input is not the same as the input"
    nsmass = edf.vmomentsurfacemass(
        0.9,
        0,
        0,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        hierarchgrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(smass - nsmass) < 0.001
    ), "surfacemass computed w/ and w/o returnGrid are not the same"
    return None


def test_mildnonaxi_meanvt_hierarchgrid_tlist():
    # Test that for a close to axisymmetric potential, the mean vt is close to that of the initial DF
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvt, grid = edf.meanvT(
        0.9,
        t=[0.0, -2.5, -5.0, -7.5, -10.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        hierarchgrid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert numpy.all(
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when using hierarchgrid and tlist"
    mvt = edf.meanvT(
        0.9,
        t=[0.0, -2.5, -5.0, -7.5, -10.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        gridpoints=_GRIDPOINTS,
    )
    assert numpy.all(
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when calculated with pre-computed grid when using hierarchgrid and tlist"
    return None


def test_mildnonaxi_meanvt_hierarchgrid_zerolevels():
    # Test that for a close to axisymmetric potential, the mean vt is close to that of the initial DF
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvt, grid = edf.meanvT(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        hierarchgrid=True,
        nlevels=0,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when using hierarchgrid"
    mvt = edf.meanvT(
        0.9, phi=0.2, integrate_method="rk6_c", grid=grid, gridpoints=_GRIDPOINTS
    )
    assert (
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when calculated with pre-computed grid when using hierarchgrid"
    return None


def test_mildnonaxi_meanvt_hierarchgrid_tlist_zerolevels():
    # Test that for a close to axisymmetric potential, the mean vt is close to that of the initial DF
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvt, grid = edf.meanvT(
        0.9,
        t=[0.0, -2.5, -5.0, -7.5, -10.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        hierarchgrid=True,
        nlevels=0,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert numpy.all(
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when using hierarchgrid and tlist"
    mvt = edf.meanvT(
        0.9,
        t=[0.0, -2.5, -5.0, -7.5, -10.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        gridpoints=_GRIDPOINTS,
    )
    assert numpy.all(
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when calculated with pre-computed grid when using hierarchgrid and tlist"
    return None


def test_mildnonaxi_meanvt_grid_rmEstimates():
    # Test vmomentsurfacemass w/o having the _estimateX functions in the initial DF
    class fakeDehnen(dehnendf):  # class that removes the _estimate functions
        def __init__(self, *args, **kwargs):
            dehnendf.__init__(self, *args, **kwargs)

        _estimatemeanvR = property()
        _estimatemeanvT = property()
        _estimateSigmaR2 = property()
        _estimateSigmaT2 = property()

    idf = fakeDehnen(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvt, grid = edf.meanvT(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf"
    return None


def test_mildnonaxi_meanvt_direct():
    # Test that for a close to axisymmetric potential, the mean vt is close to that of the initial DF
    # We do this for an axisymmetric potential, bc otherwise it takes too long
    idf = dehnendf(beta=0.0)
    pot = [LogarithmicHaloPotential(normalize=1.0)]
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvt = edf.meanvT(0.9, phi=0.2, integrate_method="rk6_c", grid=False)
    assert (
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.001
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when using direct integration"
    return None


def test_mildnonaxi_sigmar2_grid():
    # Test that for a close to axisymmetric potential, the sigmaR2 is close to the value of the initial DF
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    sr2, grid = edf.sigmaR2(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    isr2 = idf.sigmaR2(0.9)
    assert (
        numpy.fabs(numpy.log(sr2) - numpy.log(isr2)) < 0.025
    ), "sigmar2 of evolveddiskdf for axisymmetric potential is not equal to that of initial DF"
    sr2 = edf.sigmaR2(
        0.9, phi=0.2, integrate_method="rk6_c", grid=grid, gridpoints=_GRIDPOINTS
    )
    assert (
        numpy.fabs(numpy.log(sr2) - numpy.log(isr2)) < 0.025
    ), "sigmar2 of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated with pre-computed grid"
    sr2 = edf.sigmaR2(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        meanvR=_maxi_meanvr,
        surfacemass=_maxi_surfacemass,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(numpy.log(sr2) - numpy.log(isr2)) < 0.025
    ), "sigmar2 of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated with pre-computed grid and meanvR,surfacemass"
    global _maxi_sigmar2
    _maxi_sigmar2 = sr2
    return None


def test_mildnonaxi_sigmar2_direct():
    # Test that for an axisymmetric potential, the sigmaR2  is close to the value of the initial DF
    # We do this for an axisymmetric potential, bc otherwise it takes too long
    idf = dehnendf(beta=0.0)
    pot = [LogarithmicHaloPotential(normalize=1.0)]
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    sr2 = edf.sigmaR2(0.9, phi=0.2, integrate_method="rk6_c", grid=False)
    isr2 = idf.sigmaR2(0.9)
    assert (
        numpy.fabs(numpy.log(sr2) - numpy.log(isr2)) < 0.025
    ), "sigmar2 of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated directly"
    return None


def test_mildnonaxi_sigmat2_grid():
    # Test that for a close to axisymmetric potential, the sigmaR2 is close to the value of the initial DF
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    st2, grid = edf.sigmaT2(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    ist2 = idf.sigmaT2(0.9)
    assert (
        numpy.fabs(numpy.log(st2) - numpy.log(ist2)) < 0.025
    ), "sigmat2 of evolveddiskdf for axisymmetric potential is not equal to that of initial DF"
    st2 = edf.sigmaT2(
        0.9, phi=0.2, integrate_method="rk6_c", grid=grid, gridpoints=_GRIDPOINTS
    )
    assert (
        numpy.fabs(numpy.log(st2) - numpy.log(ist2)) < 0.025
    ), "sigmat2 of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated with pre-computed grid"
    st2 = edf.sigmaT2(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        meanvT=_maxi_meanvt,
        surfacemass=_maxi_surfacemass,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(numpy.log(st2) - numpy.log(ist2)) < 0.025
    ), "sigmat2 of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated with pre-computed grid and meanvR,surfacemass"
    global _maxi_sigmat2
    _maxi_sigmat2 = st2
    return None


def test_mildnonaxi_sigmat2_direct():
    # Test that for an axisymmetric potential, the sigmaT2  is close to the value of the initial DF
    # We do this for an axisymmetric potential, bc otherwise it takes too long
    idf = dehnendf(beta=0.0)
    pot = [LogarithmicHaloPotential(normalize=1.0)]
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    st2 = edf.sigmaT2(0.9, phi=0.2, integrate_method="rk6_c", grid=False)
    ist2 = idf.sigmaT2(0.9)
    assert (
        numpy.fabs(numpy.log(st2) - numpy.log(ist2)) < 0.025
    ), "sigmat2 of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated directly"
    return None


def test_mildnonaxi_sigmart_grid():
    # Test that for a close to axisymmetric potential, the sigmaR2 is close to zero
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    srt, grid = edf.sigmaRT(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(srt) < 0.01
    ), "sigmart of evolveddiskdf for axisymmetric potential is not equal to zero"
    srt = edf.sigmaRT(
        0.9, phi=0.2, integrate_method="rk6_c", grid=grid, gridpoints=_GRIDPOINTS
    )
    assert (
        numpy.fabs(srt) < 0.01
    ), "sigmart of evolveddiskdf for axisymmetric potential is not equal zero when calculated with pre-computed grid"
    srt = edf.sigmaRT(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        meanvR=_maxi_meanvr,
        meanvT=_maxi_meanvt,
        surfacemass=_maxi_surfacemass,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(srt) < 0.01
    ), "sigmart of evolveddiskdf for axisymmetric potential is not equal to zero when calculated with pre-computed grid and meanvR,surfacemass"
    global _maxi_sigmart
    _maxi_sigmart = srt
    return None


def test_mildnonaxi_sigmart_direct():
    # Test that for an axisymmetric potential, the sigmaRT is close zero
    # We do this for an axisymmetric potential, bc otherwise it takes too long
    idf = dehnendf(beta=0.0)
    pot = [LogarithmicHaloPotential(normalize=1.0)]
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    srt = edf.sigmaRT(0.9, phi=0.2, integrate_method="rk6_c", grid=False)
    assert (
        numpy.fabs(srt) < 0.01
    ), "sigmart of evolveddiskdf for axisymmetric potential is not equal to zero when calculated directly"
    return None


def test_mildnonaxi_vertexdev_grid():
    # Test that for a close to axisymmetric potential, the vertex deviation is close to zero
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    vdev, grid = edf.vertexdev(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(vdev) < 2.0 / 180.0 * numpy.pi
    ), (
        "vertexdev of evolveddiskdf for axisymmetric potential is not close to zero"
    )  # 2 is pretty big, but the weak spiral creates that
    vdev = edf.vertexdev(
        0.9, phi=0.2, integrate_method="rk6_c", grid=grid, gridpoints=_GRIDPOINTS
    )
    assert (
        numpy.fabs(vdev) < 2.0 / 180.0 * numpy.pi
    ), "vertexdev of evolveddiskdf for axisymmetric potential is not equal zero when calculated with pre-computed grid"
    vdev = edf.vertexdev(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        sigmaR2=_maxi_sigmar2,
        sigmaT2=_maxi_sigmat2,
        sigmaRT=_maxi_sigmart,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(vdev) < 2.0 / 180.0 * numpy.pi
    ), "sigmart of evolveddiskdf for axisymmetric potential is not equal to zero when calculated with pre-computed sigmaR2,sigmaT2,sigmaRT"
    return None


def test_mildnonaxi_vertexdev_direct():
    # Test that for an axisymmetric potential, the vertex deviation is close zero
    # We do this for an axisymmetric potential, bc otherwise it takes too long
    idf = dehnendf(beta=0.0)
    pot = [LogarithmicHaloPotential(normalize=1.0)]
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    vdev = edf.vertexdev(0.9, phi=0.2, integrate_method="rk6_c", grid=False)
    assert (
        numpy.fabs(vdev) < 0.01 / 180.0 * numpy.pi
    ), "vertexdev of evolveddiskdf for axisymmetric potential is not equal to zero when calculated directly"
    return None


def test_mildnonaxi_oortA_grid():
    # Test that for a close to axisymmetric potential, the oortA is close to the value of the initial DF
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(twophio=0.001),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    oa, grid, dgridR, dgridphi = edf.oortA(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        derivRGrid=True,
        derivphiGrid=True,
        returnGrids=True,
        gridpoints=_GRIDPOINTS,
        derivGridpoints=_GRIDPOINTS,
    )
    ioa = idf.oortA(0.9)
    assert (
        numpy.fabs(oa - ioa) < 0.005
    ), "oortA of evolveddiskdf for axisymmetric potential is not equal to that of initial DF"
    oa = edf.oortA(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        derivRGrid=dgridR,
        derivphiGrid=dgridphi,
        gridpoints=_GRIDPOINTS,
        derivGridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(oa - ioa) < 0.005
    ), "oortA of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated with pre-computed grid"
    return None


def test_mildnonaxi_oortA_grid_tlist():
    # Test that for a close to axisymmetric potential, the oortA is close to the value of the initial DF
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(twophio=0.001),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    oa, grid, dgridR, dgridphi = edf.oortA(
        0.9,
        t=[0.0, -2.5, -5.0, -7.5, -10.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        derivRGrid=True,
        derivphiGrid=True,
        returnGrids=True,
        gridpoints=_GRIDPOINTS,
        derivGridpoints=_GRIDPOINTS,
    )
    ioa = idf.oortA(0.9)
    assert numpy.all(
        numpy.fabs(oa - ioa) < 0.005
    ), "oortA of evolveddiskdf for axisymmetric potential is not equal to that of initial DF"
    oa = edf.oortA(
        0.9,
        t=[0.0, -2.5, -5.0, -7.5, -10.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        derivRGrid=dgridR,
        derivphiGrid=dgridphi,
        gridpoints=_GRIDPOINTS,
        derivGridpoints=_GRIDPOINTS,
    )
    assert numpy.all(
        numpy.fabs(oa - ioa) < 0.005
    ), "oortA of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated with pre-computed grid"
    return None


def test_mildnonaxi_oortB_grid():
    # Test that for a close to axisymmetric potential, the oortB is close to the value of the initial DF
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(twophio=0.001),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    ob, grid, dgridR, dgridphi = edf.oortB(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        derivRGrid=True,
        derivphiGrid=True,
        returnGrids=True,
        gridpoints=_GRIDPOINTS,
        derivGridpoints=_GRIDPOINTS,
    )
    iob = idf.oortB(0.9)
    assert (
        numpy.fabs(ob - iob) < 0.005
    ), "oortB of evolveddiskdf for axisymmetric potential is not equal to that of initial DF"
    ob = edf.oortB(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        derivRGrid=dgridR,
        derivphiGrid=dgridphi,
        gridpoints=_GRIDPOINTS,
        derivGridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(ob - iob) < 0.005
    ), "oortB of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated with pre-computed grid"
    return None


def test_mildnonaxi_oortC_grid():
    # Test that for a close to axisymmetric potential, the oortC is close to zero
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(twophio=0.001),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    oc, grid, dgridR, dgridphi = edf.oortC(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        derivRGrid=True,
        derivphiGrid=True,
        returnGrids=True,
        gridpoints=_GRIDPOINTS,
        derivGridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(oc) < 0.005
    ), "oortC of evolveddiskdf for axisymmetric potential is not equal to that of initial DF"
    oc = edf.oortC(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        derivRGrid=dgridR,
        derivphiGrid=dgridphi,
        gridpoints=_GRIDPOINTS,
        derivGridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(oc) < 0.005
    ), "oortC of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated with pre-computed grid"
    return None


def test_mildnonaxi_oortK_grid():
    # Test that for a close to axisymmetric potential, the oortK is close to zero
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(twophio=0.001),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    ok, grid, dgridR, dgridphi = edf.oortK(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        derivRGrid=True,
        derivphiGrid=True,
        returnGrids=True,
        gridpoints=_GRIDPOINTS,
        derivGridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(ok) < 0.005
    ), "oortK of evolveddiskdf for axisymmetric potential is not equal to that of initial DF"
    ok = edf.oortK(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        derivRGrid=dgridR,
        derivphiGrid=dgridphi,
        gridpoints=_GRIDPOINTS,
        derivGridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(ok) < 0.005
    ), "oortK of evolveddiskdf for axisymmetric potential is not equal to that of initial DF when calculated with pre-computed grid"
    return None


# Some special cases
def test_mildnonaxi_meanvt_grid_tlist_onet():
    # Test that for a close to axisymmetric potential, the mean vt is close to that of the initial DF, for a list consisting of a single time
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvt, grid = edf.meanvT(
        0.9,
        t=[0.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf"
    mvt = edf.meanvT(
        0.9,
        t=[0.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=grid,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvt - idf.meanvT(0.9)) < 0.005
    ), "meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when calculated with pre-computed grid"
    global _maxi_meanvt
    _maxi_meanvt = mvt
    return None


def test_mildnonaxi_meanvt_direct_tlist():
    # Shouldn't work
    idf = dehnendf(beta=0.0)
    pot = [LogarithmicHaloPotential(normalize=1.0)]
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    try:
        edf.meanvT(
            0.9,
            t=[0.0, -2.5, -5.0, -7.5, -10.0],
            phi=0.2,
            integrate_method="rk6_c",
            grid=False,
        )
    except OSError:
        pass
    else:
        raise AssertionError(
            "direct evolveddiskdf calculation of meanvT w/ list of times did not raise IOError"
        )
    return None


# Tests with significant nonaxi, but cold


def test_elliptical_cold_vr():
    # Test that the radial velocity for the elliptical disk behaves as analytically expected
    idf = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.0125))
    cp = 0.05
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(cp=cp, sp=0.0, p=0.0, tform=-150.0, tsteady=125.0),
    ]
    edf = evolveddiskdf(idf, pot=pot, to=-150.0)
    # Should be cp
    mvr, grid = edf.meanvR(
        0.9,
        phi=-numpy.pi / 4.0,
        integrate_method="rk6_c",
        grid=True,
        nsigma=7.0,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvr - cp) < 10.0**-4.0
    ), "Cold elliptical disk does not agree with analytical calculation for vr"
    # Should be 0
    mvr, grid = edf.meanvR(
        0.9,
        phi=0.0,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvr) < 10.0**-4.0
    ), "Cold elliptical disk does not agree with analytical calculation for vr"
    return None


def test_elliptical_cold_vt():
    # Test that the rotational velocity for the elliptical disk behaves as analytically expected
    idf = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.0125))
    cp = 0.05
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(cp=cp, sp=0.0, p=0.0, tform=-150.0, tsteady=125.0),
    ]
    edf = evolveddiskdf(idf, pot=pot, to=-150.0)
    # Should be 1.
    mvt, grid = edf.meanvT(
        0.9,
        phi=-numpy.pi / 4.0,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvt - 1.0) < 10.0**-3.0
    ), "Cold elliptical disk does not agree with analytical calculation for vt"
    # Should be 1.-cp
    mvt, grid = edf.meanvT(
        0.9,
        phi=0.0,
        integrate_method="rk6_c",
        grid=True,
        nsigma=7.0,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(mvt - 1.0 + cp) < 10.0**-3.0
    ), "Cold elliptical disk does not agree with analytical calculation for vt"
    return None


def test_elliptical_cold_vertexdev():
    # Test that the vertex deviations for the elliptical disk behaves as analytically expected
    idf = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.0125))
    cp = 0.05
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(cp=cp, sp=0.0, p=0.0, tform=-150.0, tsteady=125.0),
    ]
    edf = evolveddiskdf(idf, pot=pot, to=-150.0)
    # Should be -2cp in radians
    vdev, grid = edf.vertexdev(
        0.9,
        phi=-numpy.pi / 4.0,
        integrate_method="rk6_c",
        grid=True,
        nsigma=7.0,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(vdev + 2.0 * cp) < 10.0**-3.0
    ), "Cold elliptical disk does not agree with analytical calculation for vertexdev"
    # Should be 0
    vdev, grid = edf.vertexdev(
        0.9,
        phi=0.0,
        integrate_method="rk6_c",
        grid=True,
        nsigma=7.0,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    assert (
        numpy.fabs(vdev) < 10.0**-2.0 / 180.0 * numpy.pi
    ), "Cold elliptical disk does not agree with analytical calculation for vertexdev"
    return None


def test_elliptical_cold_oortABCK_position1():
    # Test that the Oort functions A, B, C, and K for the elliptical disk behaves as analytically expected
    idf = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.0125))
    cp = 0.05
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(cp=cp, sp=0.0, p=0.0, tform=-150.0, tsteady=125.0),
    ]
    edf = evolveddiskdf(idf, pot=pot, to=-150.0)
    # Should be 0.5/0.9
    oorta, grid, gridr, gridp = edf.oortA(
        0.9,
        phi=-numpy.pi / 4.0,
        integrate_method="rk6_c",
        grid=True,
        nsigma=7.0,
        derivRGrid=True,
        derivphiGrid=True,
        returnGrids=True,
        gridpoints=51,
        derivGridpoints=51,
    )
    assert (
        numpy.fabs(oorta - 0.5 / 0.9) < 10.0**-3.0
    ), "Cold elliptical disk does not agree with analytical calculation for oortA"
    # Also check other Oort constants
    # Should be -0.5/0.9
    oortb = edf.oortB(
        0.9,
        phi=-numpy.pi / 4.0,
        integrate_method="rk6_c",
        grid=grid,
        nsigma=7.0,
        derivRGrid=gridr,
        derivphiGrid=gridp,
    )
    assert (
        numpy.fabs(oortb + 0.5 / 0.9) < 10.0**-3.0
    ), "Cold elliptical disk does not agree with analytical calculation for oortB"
    # Should be cp/2
    oortc = edf.oortC(
        0.9,
        phi=-numpy.pi / 4.0,
        integrate_method="rk6_c",
        grid=grid,
        nsigma=7.0,
        derivRGrid=gridr,
        derivphiGrid=gridp,
    )
    assert (
        numpy.fabs(oortc - cp / 2.0) < 10.0**-2.2
    ), "Cold elliptical disk does not agree with analytical calculation for oortC"
    # Should be -cp/2
    oortk = edf.oortK(
        0.9,
        phi=-numpy.pi / 4.0,
        integrate_method="rk6_c",
        grid=grid,
        nsigma=7.0,
        derivRGrid=gridr,
        derivphiGrid=gridp,
    )
    assert (
        numpy.fabs(oortk + cp / 2.0) < 10.0**-2.2
    ), "Cold elliptical disk does not agree with analytical calculation for oortK"
    return None


def test_elliptical_cold_oortABCK_position2():
    # Test that the Oort functions A, B, C, and K for the elliptical disk behaves as analytically expected
    idf = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.0125))
    cp = 0.05
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(cp=cp, sp=0.0, p=0.0, tform=-150.0, tsteady=125.0),
    ]
    edf = evolveddiskdf(idf, pot=pot, to=-150.0)
    # Should be 0.5/0.9+cp/2
    oorta, grid, gridr, gridp = edf.oortA(
        0.9,
        phi=0.0,
        integrate_method="rk6_c",
        grid=True,
        nsigma=7.0,
        derivRGrid=True,
        derivphiGrid=True,
        returnGrids=True,
        gridpoints=51,
        derivGridpoints=51,
    )
    assert (
        numpy.fabs(oorta - cp / 2.0 - 0.5 / 0.9) < 10.0**-2.2
    ), "Cold elliptical disk does not agree with analytical calculation for oortA"
    # Should be -cp/2-0.5/0.9
    oortb = edf.oortB(
        0.9,
        phi=0.0,
        integrate_method="rk6_c",
        grid=grid,
        nsigma=7.0,
        derivRGrid=gridr,
        derivphiGrid=gridp,
    )
    assert (
        numpy.fabs(oortb + cp / 2.0 + 0.5 / 0.9) < 10.0**-2.2
    ), "Cold elliptical disk does not agree with analytical calculation for oortB"
    # Should be 0
    oortc = edf.oortC(
        0.9,
        phi=0.0,
        integrate_method="rk6_c",
        grid=grid,
        nsigma=7.0,
        derivRGrid=gridr,
        derivphiGrid=gridp,
    )
    assert (
        numpy.fabs(oortc) < 10.0**-3.0
    ), "Cold elliptical disk does not agree with analytical calculation for oortC"
    # Should be 0
    oortk = edf.oortK(
        0.9,
        phi=0.0,
        integrate_method="rk6_c",
        grid=grid,
        nsigma=7.0,
        derivRGrid=gridr,
        derivphiGrid=gridp,
    )
    assert (
        numpy.fabs(oortk) < 10.0**-3.0
    ), "Cold elliptical disk does not agree with analytical calculation for oortK"
    return None


def test_call_special():
    from galpy.orbit import Orbit

    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(twophio=0.001),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    o = Orbit([0.9, 0.1, 1.1, 2.0])
    # call w/ and w/o explicit t
    assert (
        numpy.fabs(numpy.log(edf(o, 0.0)) - numpy.log(edf(o))) < 10.0**-10.0
    ), "edf.__call__ w/ explicit t=0. and w/o t do not give the same answer"
    # call must get Orbit, otherwise error
    try:
        edf(0.9, 0.1, 1.1, 2.0)
    except OSError:
        pass
    else:
        raise AssertionError("edf.__call__ w/o Orbit input did not raise IOError")
    # Call w/ list, but just to
    assert (
        numpy.fabs(numpy.log(edf(o, [-10.0])) - numpy.log(idf(o))) < 10.0**-10.0
    ), "edf.__call__ w/ tlist set to [to] did not return initial DF"
    # Call w/ just to
    assert (
        numpy.fabs(numpy.log(edf(o, -10.0)) - numpy.log(idf(o))) < 10.0**-10.0
    ), "edf.__call__ w/ tlist set to [to] did not return initial DF"
    # also w/ log
    assert (
        numpy.fabs(edf(o, [-10.0], log=True) - numpy.log(idf(o))) < 10.0**-10.0
    ), "edf.__call__ w/ tlist set to [to] did not return initial DF (log)"
    assert (
        numpy.fabs(edf(o, -10.0, log=True) - numpy.log(idf(o))) < 10.0**-10.0
    ), "edf.__call__ w/ tlist set to [to] did not return initial DF (log)"
    # Tests w/ odeint: tlist, one NaN
    codeint = edf(
        o, [0.0, -2.5, -5.0, -7.5, -10.0], integrate_method="odeint", log=True
    )
    crk6c = edf(o, [0.0, -2.5, -5.0, -7.5, -10.0], integrate_method="rk6_c", log=True)
    assert numpy.all(
        numpy.fabs(codeint - crk6c) < 10.0**-4.0
    ), "edf.__call__ w/ odeint and tlist does not give the same result as w/ rk6_c"
    # Crazy orbit w/ tlist
    crk6c = edf(
        Orbit([3.0, 1.0, -1.0, 2.0]), [0.0], integrate_method="odeint", log=True
    )
    assert crk6c < -20.0, "crazy orbit does not have DF equal to zero"
    # deriv w/ odeint
    codeint = edf(
        o, [0.0, -2.5, -5.0, -7.5, -10.0], integrate_method="odeint", deriv="R"
    )
    crk6c = edf(o, [0.0, -2.5, -5.0, -7.5, -10.0], integrate_method="rk6_c", deriv="R")
    assert numpy.all(
        numpy.fabs(codeint - crk6c) < 10.0**-4.0
    ), "edf.__call__ w/ odeint and tlist does not give the same result as w/ rk6_c (deriv=R)"
    # deriv w/ len(tlist)=1
    crk6c = edf(o, [0.0], integrate_method="rk6_c", deriv="R")
    crk6c2 = edf(o, 0.0, integrate_method="rk6_c", deriv="R")
    assert numpy.all(
        numpy.fabs(crk6c - crk6c2) < 10.0**-4.0
    ), "edf.__call__ w/ tlist consisting of one time and just a scalar time do not agree"
    # Call w/ just to and deriv
    assert (
        numpy.fabs(
            edf(o, -10.0, deriv="R")
            - idf(o) * idf._dlnfdR(o.vxvv[0, 0], o.vxvv[0, 1], o.vxvv[0, 2])
        )
        < 10.0**-10.0
    ), "edf.__call__ w/ to did not return initial DF (deriv=R)"
    assert (
        numpy.fabs(edf(o, -10.0, deriv="phi")) < 10.0**-10.0
    ), "edf.__call__ w/ to did not return initial DF (deriv=phi)"
    # Call w/ just one t and odeint
    codeint = edf(o, 0, integrate_method="odeint", log=True)
    crk6c = edf(o, 0.0, integrate_method="rk6_c", log=True)
    assert (
        numpy.fabs(codeint - crk6c) < 10.0**-4.0
    ), "edf.__call__ w/ odeint and tlist does not give the same result as w/ rk6_c"
    # Call w/ just one t and fallback to odeint
    # turn off C
    edf._pot[0].hasC = False
    edf._pot[0].hasC_dxdv = False
    codeint = edf(o, 0, integrate_method="dopr54_c", log=True)
    assert (
        numpy.fabs(codeint - crk6c) < 10.0**-4.0
    ), "edf.__call__ w/ odeint and tlist does not give the same result as w/ rk6_c"
    # Call w/ just one t and fallback to leaprog
    cleapfrog = edf(o, 0, integrate_method="leapfrog_c", log=True)
    assert (
        numpy.fabs(cleapfrog - crk6c) < 10.0**-4.0
    ), "edf.__call__ w/ odeint and tlist does not give the same result as w/ rk6_c"
    # Call w/ just one t agrees whether or not t is list
    cleapfrog_list = edf(o, [-2.5], integrate_method="leapfrog_c", log=True)
    cleapfrog_scal = edf(o, -2.5, integrate_method="leapfrog_c", log=True)
    assert (
        numpy.fabs(cleapfrog_list - cleapfrog_scal) < 10.0**-4.0
    ), "edf.__call__ w/ single t scalar or tlist does not give the same result"
    # Radial orbit
    o = Orbit([1.0, -1.0, 0.0, 0.0])
    assert (
        numpy.fabs(edf(o, 0.0)) < 10.0**-10.0
    ), "edf.__call__ w/ radial orbit does not return zero"


def test_call_marginalizevperp():
    from galpy.orbit import Orbit

    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(twophio=0.001),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot[0], to=-10.0)  # one with just one potential
    # l=0
    R, phi, vR = 0.8, 0.0, 0.4
    vts = numpy.linspace(0.0, 1.5, 51)
    pvts = numpy.array(
        [edf(Orbit([R, vR, vt, phi]), integrate_method="rk6_c") for vt in vts]
    )
    assert (
        numpy.fabs(
            numpy.sum(pvts) * (vts[1] - vts[0])
            - edf(
                Orbit([R, vR, 0.0, phi]),
                marginalizeVperp=True,
                integrate_method="rk6_c",
            )
        )
        < 10.0**-3.5
    ), "evolveddiskdf call w/ marginalizeVperp does not work"
    # l=270
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    R, phi, vT = numpy.sin(numpy.pi / 6.0), -numpy.pi / 3.0, 0.7  # l=30 degree
    vrs = numpy.linspace(-1.0, 1.0, 101)
    pvrs = numpy.array(
        [edf(Orbit([R, vr, vT, phi]), integrate_method="rk6_c") for vr in vrs]
    )
    assert (
        numpy.fabs(
            numpy.log(numpy.sum(pvrs) * (vrs[1] - vrs[0]))
            - edf(
                Orbit([R, 0.0, vT, phi]),
                marginalizeVperp=True,
                integrate_method="rk6_c",
                log=True,
                nsigma=4,
            )
        )
        < 10.0**-2.5
    ), "evolveddiskdf call w/ marginalizeVperp does not work"
    return None


def test_call_marginalizevlos():
    from galpy.orbit import Orbit

    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        EllipticalDiskPotential(twophio=0.001),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot[0], to=-10.0)  # one with just one potential
    # l=0
    R, phi, vT = 0.8, 0.0, 0.7
    vrs = numpy.linspace(-1.0, 1.0, 101)
    pvrs = numpy.array(
        [edf(Orbit([R, vr, vT, phi]), integrate_method="rk6_c") for vr in vrs]
    )
    assert (
        numpy.fabs(
            numpy.log(numpy.sum(pvrs) * (vrs[1] - vrs[0]))
            - edf(
                Orbit([R, 0.0, vT, phi]),
                marginalizeVlos=True,
                integrate_method="rk6_c",
                log=True,
            )
        )
        < 10.0**-4.0
    ), "diskdf call w/ marginalizeVlos does not work"
    # l=270, this DF has some issues, but it suffices to test the mechanics of the code
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    R, phi, vR = numpy.sin(numpy.pi / 6.0), -numpy.pi / 3.0, 0.4  # l=30 degree
    vts = numpy.linspace(0.3, 1.5, 101)
    pvts = numpy.array(
        [edf(Orbit([R, vR, vt, phi]), integrate_method="rk6_c") for vt in vts]
    )
    assert (
        numpy.fabs(
            numpy.sum(pvts) * (vts[1] - vts[0])
            - edf(
                Orbit([R, vR, 0.0, phi]),
                marginalizeVlos=True,
                integrate_method="rk6_c",
                nsigma=4,
            )
        )
        < 10.0**-3.5
    ), "diskdf call w/ marginalizeVlos does not work"
    return None


def test_plot_grid():
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvr, grid = edf.meanvR(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    grid.plot()
    # w/ list of times
    mvr, grid = edf.meanvR(
        0.9,
        t=[0.0, -2.5, -5.0, -7.5, -10.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    grid.plot(1)
    return None


def test_plot_hierarchgrid():
    idf = dehnendf(beta=0.0)
    pot = [
        LogarithmicHaloPotential(normalize=1.0),
        SteadyLogSpiralPotential(A=-0.005, omegas=0.2),
    ]  # very mild non-axi
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    mvr, grid = edf.meanvR(
        0.9,
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        hierarchgrid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    grid.plot()
    # w/ list of times
    mvr, grid = edf.meanvR(
        0.9,
        t=[0.0, -2.5, -5.0, -7.5, -10.0],
        phi=0.2,
        integrate_method="rk6_c",
        grid=True,
        hierarchgrid=True,
        returnGrid=True,
        gridpoints=_GRIDPOINTS,
    )
    grid.plot(1)
    return None
