# Tests of the quasiisothermaldf module
import numpy

from galpy.actionAngle import actionAngleAdiabatic, actionAngleStaeckel
from galpy.df import quasiisothermaldf

# fiducial setup uses these
from galpy.potential import MWPotential, epifreq, omegac, vcirc, verticalfreq

aAA = actionAngleAdiabatic(pot=MWPotential, c=True)
aAS = actionAngleStaeckel(pot=MWPotential, c=True, delta=0.5)


def test_pvRvT_adiabatic():
    qdf = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aAA, cutcounter=True
    )
    R, z = 0.8, 0.1
    vRs = numpy.linspace(-1.0, 1.0, 21)
    vTs = numpy.linspace(0.0, 1.5, 51)
    pvRvT = numpy.array([[qdf.pvRvT(vr, vt, R, z) for vt in vTs] for vr in vRs])
    tvR = numpy.tile(vRs, (len(vTs), 1)).T
    tvT = numpy.tile(vTs, (len(vRs), 1))
    mvR = numpy.sum(tvR * pvRvT) / numpy.sum(pvRvT)
    mvT = numpy.sum(tvT * pvRvT) / numpy.sum(pvRvT)
    svR = numpy.sqrt(numpy.sum(tvR**2.0 * pvRvT) / numpy.sum(pvRvT) - mvR**2.0)
    svT = numpy.sqrt(numpy.sum(tvT**2.0 * pvRvT) / numpy.sum(pvRvT) - mvT**2.0)
    svRvT = (numpy.sum(tvR * tvT * pvRvT) / numpy.sum(pvRvT) - mvR * mvT) / svR / svT
    assert (
        numpy.fabs(mvR) < 0.01
    ), "mean vR calculated from pvRvT not equal to zero for adiabatic actions"
    assert (
        numpy.fabs(mvT - qdf.meanvT(R, z)) < 0.01
    ), "mean vT calculated from pvRvT not equal to zero for adiabatic actions"
    assert (
        numpy.fabs(numpy.log(svR) - 0.5 * numpy.log(qdf.sigmaR2(R, z))) < 0.01
    ), "sigma vR calculated from pvRvT not equal to that from sigmaR2 for adiabatic actions"
    assert (
        numpy.fabs(numpy.log(svT) - 0.5 * numpy.log(qdf.sigmaT2(R, z))) < 0.01
    ), "sigma vT calculated from pvRvT not equal to that from sigmaT2 for adiabatic actions"
    assert (
        numpy.fabs(svRvT) < 0.01
    ), "correlation between vR and vT calculated from pvRvT not equal to zero for adiabatic actions"
    return None


def test_pvRvT_staeckel():
    qdf = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aAS, cutcounter=True
    )
    R, z = 0.8, 0.1
    vRs = numpy.linspace(-1.0, 1.0, 21)
    vTs = numpy.linspace(0.0, 1.5, 51)
    pvRvT = numpy.array([[qdf.pvRvT(vr, vt, R, z) for vt in vTs] for vr in vRs])
    tvR = numpy.tile(vRs, (len(vTs), 1)).T
    tvT = numpy.tile(vTs, (len(vRs), 1))
    mvR = numpy.sum(tvR * pvRvT) / numpy.sum(pvRvT)
    mvT = numpy.sum(tvT * pvRvT) / numpy.sum(pvRvT)
    svR = numpy.sqrt(numpy.sum(tvR**2.0 * pvRvT) / numpy.sum(pvRvT) - mvR**2.0)
    svT = numpy.sqrt(numpy.sum(tvT**2.0 * pvRvT) / numpy.sum(pvRvT) - mvT**2.0)
    svRvT = (numpy.sum(tvR * tvT * pvRvT) / numpy.sum(pvRvT) - mvR * mvT) / svR / svT
    assert (
        numpy.fabs(mvR) < 0.01
    ), "mean vR calculated from pvRvT not equal to zero for staeckel actions"
    assert (
        numpy.fabs(mvT - qdf.meanvT(R, z)) < 0.01
    ), "mean vT calculated from pvRvT not equal to zero for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svR) - 0.5 * numpy.log(qdf.sigmaR2(R, z))) < 0.01
    ), "sigma vR calculated from pvRvT not equal to that from sigmaR2 for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svT) - 0.5 * numpy.log(qdf.sigmaT2(R, z))) < 0.01
    ), "sigma vT calculated from pvRvT not equal to that from sigmaT2 for staeckel actions"
    assert (
        numpy.fabs(svRvT) < 0.01
    ), "correlation between vR and vT calculated from pvRvT not equal to zero for staeckel actions"
    return None


def test_pvRvT_staeckel_diffngl():
    qdf = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aAS, cutcounter=True
    )
    R, z = 0.8, 0.1
    vRs = numpy.linspace(-1.0, 1.0, 21)
    vTs = numpy.linspace(0.0, 1.5, 51)
    # ngl=10
    pvRvT = numpy.array([[qdf.pvRvT(vr, vt, R, z, ngl=10) for vt in vTs] for vr in vRs])
    tvR = numpy.tile(vRs, (len(vTs), 1)).T
    tvT = numpy.tile(vTs, (len(vRs), 1))
    mvR = numpy.sum(tvR * pvRvT) / numpy.sum(pvRvT)
    mvT = numpy.sum(tvT * pvRvT) / numpy.sum(pvRvT)
    svR = numpy.sqrt(numpy.sum(tvR**2.0 * pvRvT) / numpy.sum(pvRvT) - mvR**2.0)
    svT = numpy.sqrt(numpy.sum(tvT**2.0 * pvRvT) / numpy.sum(pvRvT) - mvT**2.0)
    svRvT = (numpy.sum(tvR * tvT * pvRvT) / numpy.sum(pvRvT) - mvR * mvT) / svR / svT
    assert (
        numpy.fabs(mvR) < 0.01
    ), "mean vR calculated from pvRvT not equal to zero for staeckel actions"
    assert (
        numpy.fabs(mvT - qdf.meanvT(R, z)) < 0.01
    ), "mean vT calculated from pvRvT not equal to zero for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svR) - 0.5 * numpy.log(qdf.sigmaR2(R, z))) < 0.01
    ), "sigma vR calculated from pvRvT not equal to that from sigmaR2 for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svT) - 0.5 * numpy.log(qdf.sigmaT2(R, z))) < 0.01
    ), "sigma vT calculated from pvRvT not equal to that from sigmaT2 for staeckel actions"
    assert (
        numpy.fabs(svRvT) < 0.01
    ), "correlation between vR and vT calculated from pvRvT not equal to zero for staeckel actions"
    # ngl=24
    pvRvT = numpy.array([[qdf.pvRvT(vr, vt, R, z, ngl=40) for vt in vTs] for vr in vRs])
    mvR = numpy.sum(tvR * pvRvT) / numpy.sum(pvRvT)
    mvT = numpy.sum(tvT * pvRvT) / numpy.sum(pvRvT)
    svR = numpy.sqrt(numpy.sum(tvR**2.0 * pvRvT) / numpy.sum(pvRvT) - mvR**2.0)
    svT = numpy.sqrt(numpy.sum(tvT**2.0 * pvRvT) / numpy.sum(pvRvT) - mvT**2.0)
    svRvT = (numpy.sum(tvR * tvT * pvRvT) / numpy.sum(pvRvT) - mvR * mvT) / svR / svT
    assert (
        numpy.fabs(mvR) < 0.01
    ), "mean vR calculated from pvRvT not equal to zero for staeckel actions"
    assert (
        numpy.fabs(mvT - qdf.meanvT(R, z)) < 0.01
    ), "mean vT calculated from pvRvT not equal to zero for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svR) - 0.5 * numpy.log(qdf.sigmaR2(R, z))) < 0.01
    ), "sigma vR calculated from pvRvT not equal to that from sigmaR2 for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svT) - 0.5 * numpy.log(qdf.sigmaT2(R, z))) < 0.01
    ), "sigma vT calculated from pvRvT not equal to that from sigmaT2 for staeckel actions"
    assert (
        numpy.fabs(svRvT) < 0.01
    ), "correlation between vR and vT calculated from pvRvT not equal to zero for staeckel actions"
    # ngl=11, shouldn't work
    try:
        pvRvT = numpy.array(
            [[qdf.pvRvT(vr, vt, R, z, ngl=11) for vt in vTs] for vr in vRs]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("pvz w/ ngl=odd did not raise ValueError")
    return None


def test_pvTvz_adiabatic():
    qdf = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aAA, cutcounter=True
    )
    R, z = 0.8, 0.1
    vTs = numpy.linspace(0.0, 1.5, 51)
    vzs = numpy.linspace(-1.0, 1.0, 21)
    pvTvz = numpy.array([[qdf.pvTvz(vt, vz, R, z) for vt in vTs] for vz in vzs])
    tvT = numpy.tile(vTs, (len(vzs), 1))
    tvz = numpy.tile(vzs, (len(vTs), 1)).T
    mvz = numpy.sum(tvz * pvTvz) / numpy.sum(pvTvz)
    mvT = numpy.sum(tvT * pvTvz) / numpy.sum(pvTvz)
    svz = numpy.sqrt(numpy.sum(tvz**2.0 * pvTvz) / numpy.sum(pvTvz) - mvz**2.0)
    svT = numpy.sqrt(numpy.sum(tvT**2.0 * pvTvz) / numpy.sum(pvTvz) - mvT**2.0)
    svTvz = (numpy.sum(tvz * tvT * pvTvz) / numpy.sum(pvTvz) - mvz * mvT) / svz / svT
    assert (
        numpy.fabs(mvz) < 0.01
    ), "mean vz calculated from pvTvz not equal to zero for adiabatic actions"
    assert (
        numpy.fabs(mvT - qdf.meanvT(R, z)) < 0.01
    ), "mean vT calculated from pvTvz not equal to zero for adiabatic actions"
    assert (
        numpy.fabs(numpy.log(svz) - 0.5 * numpy.log(qdf.sigmaz2(R, z))) < 0.01
    ), "sigma vz calculated from pvTvz not equal to that from sigmaz2 for adiabatic actions"
    assert (
        numpy.fabs(numpy.log(svT) - 0.5 * numpy.log(qdf.sigmaT2(R, z))) < 0.01
    ), "sigma vT calculated from pvTvz not equal to that from sigmaT2 for adiabatic actions"
    assert (
        numpy.fabs(svTvz) < 0.01
    ), "correlation between vz and vT calculated from pvTvz not equal to zero for adiabatic actions"
    return None


def test_pvTvz_staeckel():
    qdf = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aAS, cutcounter=True
    )
    R, z = 0.8, 0.1
    vzs = numpy.linspace(-1.0, 1.0, 21)
    vTs = numpy.linspace(0.0, 1.5, 51)
    pvTvz = numpy.array([[qdf.pvTvz(vt, vz, R, z) for vt in vTs] for vz in vzs])
    tvz = numpy.tile(vzs, (len(vTs), 1)).T
    tvT = numpy.tile(vTs, (len(vzs), 1))
    mvz = numpy.sum(tvz * pvTvz) / numpy.sum(pvTvz)
    mvT = numpy.sum(tvT * pvTvz) / numpy.sum(pvTvz)
    svz = numpy.sqrt(numpy.sum(tvz**2.0 * pvTvz) / numpy.sum(pvTvz) - mvz**2.0)
    svT = numpy.sqrt(numpy.sum(tvT**2.0 * pvTvz) / numpy.sum(pvTvz) - mvT**2.0)
    svTvz = (numpy.sum(tvz * tvT * pvTvz) / numpy.sum(pvTvz) - mvz * mvT) / svz / svT
    assert (
        numpy.fabs(mvz) < 0.01
    ), "mean vz calculated from pvTvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(mvT - qdf.meanvT(R, z)) < 0.01
    ), "mean vT calculated from pvTvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svz) - 0.5 * numpy.log(qdf.sigmaz2(R, z))) < 0.01
    ), "sigma vz calculated from pvTvz not equal to that from sigmaz2 for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svT) - 0.5 * numpy.log(qdf.sigmaT2(R, z))) < 0.01
    ), "sigma vT calculated from pvTvz not equal to that from sigmaT2 for staeckel actions"
    assert (
        numpy.fabs(svTvz) < 0.01
    ), "correlation between vz and vT calculated from pvTvz not equal to zero for staeckel actions"
    return None


def test_pvTvz_staeckel_diffngl():
    qdf = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aAS, cutcounter=True
    )
    R, z = 0.8, 0.1
    vzs = numpy.linspace(-1.0, 1.0, 21)
    vTs = numpy.linspace(0.0, 1.5, 51)
    # ngl=10
    pvTvz = numpy.array([[qdf.pvTvz(vt, vz, R, z, ngl=10) for vt in vTs] for vz in vzs])
    tvz = numpy.tile(vzs, (len(vTs), 1)).T
    tvT = numpy.tile(vTs, (len(vzs), 1))
    mvz = numpy.sum(tvz * pvTvz) / numpy.sum(pvTvz)
    mvT = numpy.sum(tvT * pvTvz) / numpy.sum(pvTvz)
    svz = numpy.sqrt(numpy.sum(tvz**2.0 * pvTvz) / numpy.sum(pvTvz) - mvz**2.0)
    svT = numpy.sqrt(numpy.sum(tvT**2.0 * pvTvz) / numpy.sum(pvTvz) - mvT**2.0)
    svTvz = (numpy.sum(tvz * tvT * pvTvz) / numpy.sum(pvTvz) - mvz * mvT) / svz / svT
    assert (
        numpy.fabs(mvz) < 0.01
    ), "mean vz calculated from pvTvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(mvT - qdf.meanvT(R, z)) < 0.01
    ), "mean vT calculated from pvTvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svz) - 0.5 * numpy.log(qdf.sigmaz2(R, z))) < 0.01
    ), "sigma vz calculated from pvTvz not equal to that from sigmaz2 for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svT) - 0.5 * numpy.log(qdf.sigmaT2(R, z))) < 0.01
    ), "sigma vT calculated from pvTvz not equal to that from sigmaT2 for staeckel actions"
    assert (
        numpy.fabs(svTvz) < 0.01
    ), "correlation between vz and vT calculated from pvTvz not equal to zero for staeckel actions"
    # ngl=24
    pvTvz = numpy.array([[qdf.pvTvz(vt, vz, R, z, ngl=40) for vt in vTs] for vz in vzs])
    mvz = numpy.sum(tvz * pvTvz) / numpy.sum(pvTvz)
    mvT = numpy.sum(tvT * pvTvz) / numpy.sum(pvTvz)
    svz = numpy.sqrt(numpy.sum(tvz**2.0 * pvTvz) / numpy.sum(pvTvz) - mvz**2.0)
    svT = numpy.sqrt(numpy.sum(tvT**2.0 * pvTvz) / numpy.sum(pvTvz) - mvT**2.0)
    svTvz = (numpy.sum(tvz * tvT * pvTvz) / numpy.sum(pvTvz) - mvz * mvT) / svz / svT
    assert (
        numpy.fabs(mvz) < 0.01
    ), "mean vz calculated from pvTvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(mvT - qdf.meanvT(R, z)) < 0.01
    ), "mean vT calculated from pvTvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svz) - 0.5 * numpy.log(qdf.sigmaz2(R, z))) < 0.01
    ), "sigma vz calculated from pvTvz not equal to that from sigmaz2 for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svT) - 0.5 * numpy.log(qdf.sigmaT2(R, z))) < 0.01
    ), "sigma vT calculated from pvTvz not equal to that from sigmaT2 for staeckel actions"
    assert (
        numpy.fabs(svTvz) < 0.01
    ), "correlation between vz and vT calculated from pvTvz not equal to zero for staeckel actions"
    # ngl=11, shouldn't work
    try:
        pvTvz = numpy.array(
            [[qdf.pvTvz(vt, vz, R, z, ngl=11) for vt in vTs] for vz in vzs]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("pvz w/ ngl=odd did not raise ValueError")
    return None


def test_pvRvz_adiabatic():
    qdf = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aAA, cutcounter=True
    )
    R, z = 0.8, 0.1
    vRs = numpy.linspace(-1.0, 1.0, 21)
    vzs = numpy.linspace(-1.0, 1.0, 21)
    pvRvz = numpy.array([[qdf.pvRvz(vr, vz, R, z) for vz in vzs] for vr in vRs])
    tvR = numpy.tile(vRs, (len(vzs), 1)).T
    tvz = numpy.tile(vzs, (len(vRs), 1))
    mvR = numpy.sum(tvR * pvRvz) / numpy.sum(pvRvz)
    mvz = numpy.sum(tvz * pvRvz) / numpy.sum(pvRvz)
    svR = numpy.sqrt(numpy.sum(tvR**2.0 * pvRvz) / numpy.sum(pvRvz) - mvR**2.0)
    svz = numpy.sqrt(numpy.sum(tvz**2.0 * pvRvz) / numpy.sum(pvRvz) - mvz**2.0)
    svRvz = (numpy.sum(tvR * tvz * pvRvz) / numpy.sum(pvRvz) - mvR * mvz) / svR / svz
    sR2 = qdf.sigmaR2(R, z)  # direct calculation
    sz2 = qdf.sigmaz2(R, z)
    assert (
        numpy.fabs(mvR) < 0.01
    ), "mean vR calculated from pvRvz not equal to zero for adiabatic actions"
    assert (
        numpy.fabs(mvz) < 0.01
    ), "mean vz calculated from pvRvz not equal to zero for adiabatic actions"
    assert (
        numpy.fabs(numpy.log(svR) - 0.5 * numpy.log(sR2)) < 0.01
    ), "sigma vR calculated from pvRvz not equal to that from sigmaR2 for adiabatic actions"
    assert (
        numpy.fabs(numpy.log(svz) - 0.5 * numpy.log(sz2)) < 0.01
    ), "sigma vz calculated from pvRvz not equal to that from sigmaz2 for adiabatic actions"
    assert (
        numpy.fabs(svRvz - qdf.sigmaRz(R, z) / numpy.sqrt(sR2 * sz2)) < 0.01
    ), "correlation between vR and vz calculated from pvRvz not equal to zero for adiabatic actions"
    return None


def test_pvRvz_staeckel():
    qdf = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aAS, cutcounter=True
    )
    R, z = 0.8, 0.1
    vRs = numpy.linspace(-1.0, 1.0, 21)
    vzs = numpy.linspace(-1.0, 1.0, 21)
    pvRvz = numpy.array([[qdf.pvRvz(vr, vz, R, z) for vz in vzs] for vr in vRs])
    tvR = numpy.tile(vRs, (len(vzs), 1)).T
    tvz = numpy.tile(vzs, (len(vRs), 1))
    mvR = numpy.sum(tvR * pvRvz) / numpy.sum(pvRvz)
    mvz = numpy.sum(tvz * pvRvz) / numpy.sum(pvRvz)
    svR = numpy.sqrt(numpy.sum(tvR**2.0 * pvRvz) / numpy.sum(pvRvz) - mvR**2.0)
    svz = numpy.sqrt(numpy.sum(tvz**2.0 * pvRvz) / numpy.sum(pvRvz) - mvz**2.0)
    svRvz = (numpy.sum(tvR * tvz * pvRvz) / numpy.sum(pvRvz) - mvR * mvz) / svR / svz
    sR2 = qdf.sigmaR2(R, z)  # direct calculation
    sz2 = qdf.sigmaz2(R, z)
    assert (
        numpy.fabs(mvR) < 0.01
    ), "mean vR calculated from pvRvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(mvz) < 0.01
    ), "mean vz calculated from pvRvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svR) - 0.5 * numpy.log(sR2)) < 0.01
    ), "sigma vR calculated from pvRvz not equal to that from sigmaR2 for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svz) - 0.5 * numpy.log(sz2)) < 0.01
    ), "sigma vz calculated from pvRvz not equal to that from sigmaz2 for staeckel actions"
    assert (
        numpy.fabs(svRvz - qdf.sigmaRz(R, z) / numpy.sqrt(sR2 * sz2)) < 0.01
    ), "correlation between vR and vz calculated from pvRvz not equal to zero for adiabatic actions"
    return None


def test_pvRvz_staeckel_diffngl():
    qdf = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aAS, cutcounter=True
    )
    R, z = 0.8, 0.1
    vRs = numpy.linspace(-1.0, 1.0, 21)
    vzs = numpy.linspace(-1.0, 1.0, 21)
    # ngl=10
    pvRvz = numpy.array([[qdf.pvRvz(vr, vz, R, z, ngl=10) for vz in vzs] for vr in vRs])
    tvR = numpy.tile(vRs, (len(vzs), 1)).T
    tvz = numpy.tile(vzs, (len(vRs), 1))
    mvR = numpy.sum(tvR * pvRvz) / numpy.sum(pvRvz)
    mvz = numpy.sum(tvz * pvRvz) / numpy.sum(pvRvz)
    svR = numpy.sqrt(numpy.sum(tvR**2.0 * pvRvz) / numpy.sum(pvRvz) - mvR**2.0)
    svz = numpy.sqrt(numpy.sum(tvz**2.0 * pvRvz) / numpy.sum(pvRvz) - mvz**2.0)
    svRvz = (numpy.sum(tvR * tvz * pvRvz) / numpy.sum(pvRvz) - mvR * mvz) / svR / svz
    sR2 = qdf.sigmaR2(R, z)  # direct calculation
    sz2 = qdf.sigmaz2(R, z)
    assert (
        numpy.fabs(mvR) < 0.01
    ), "mean vR calculated from pvRvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(mvz) < 0.01
    ), "mean vz calculated from pvRvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svR) - 0.5 * numpy.log(sR2)) < 0.01
    ), "sigma vR calculated from pvRvz not equal to that from sigmaR2 for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svz) - 0.5 * numpy.log(sz2)) < 0.01
    ), "sigma vz calculated from pvRvz not equal to that from sigmaz2 for staeckel actions"
    assert (
        numpy.fabs(svRvz - qdf.sigmaRz(R, z) / numpy.sqrt(sR2 * sz2)) < 0.01
    ), "correlation between vR and vz calculated from pvRvz not equal to zero for adiabatic actions"
    # ngl=24
    pvRvz = numpy.array([[qdf.pvRvz(vr, vz, R, z, ngl=40) for vz in vzs] for vr in vRs])
    mvR = numpy.sum(tvR * pvRvz) / numpy.sum(pvRvz)
    mvz = numpy.sum(tvz * pvRvz) / numpy.sum(pvRvz)
    svR = numpy.sqrt(numpy.sum(tvR**2.0 * pvRvz) / numpy.sum(pvRvz) - mvR**2.0)
    svz = numpy.sqrt(numpy.sum(tvz**2.0 * pvRvz) / numpy.sum(pvRvz) - mvz**2.0)
    svRvz = (numpy.sum(tvR * tvz * pvRvz) / numpy.sum(pvRvz) - mvR * mvz) / svR / svz
    assert (
        numpy.fabs(mvR) < 0.01
    ), "mean vR calculated from pvRvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(mvz) < 0.01
    ), "mean vz calculated from pvRvz not equal to zero for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svR) - 0.5 * numpy.log(sR2)) < 0.01
    ), "sigma vR calculated from pvRvz not equal to that from sigmaR2 for staeckel actions"
    assert (
        numpy.fabs(numpy.log(svz) - 0.5 * numpy.log(sz2)) < 0.01
    ), "sigma vz calculated from pvRvz not equal to that from sigmaz2 for staeckel actions"
    assert (
        numpy.fabs(svRvz - qdf.sigmaRz(R, z) / numpy.sqrt(sR2 * sz2)) < 0.01
    ), "correlation between vR and vz calculated from pvRvz not equal to zero for adiabatic actions"
    # ngl=11, shouldn't work
    try:
        pvRvz = numpy.array(
            [[qdf.pvRvz(vr, vz, R, z, ngl=11) for vz in vzs] for vr in vRs]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("pvz w/ ngl=odd did not raise ValueError")
    return None


def test_pvRvz_staeckel_arrayin():
    qdf = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aAS, cutcounter=True
    )
    R, z = 0.8, 0.1
    pvRvz = qdf.pvRvz(
        0.1 * numpy.ones(2), 0.05 * numpy.ones(2), R * numpy.ones(2), z * numpy.ones(2)
    )
    assert numpy.all(
        numpy.fabs(numpy.log(pvRvz) - numpy.log(qdf.pvRvz(0.1, 0.05, R, z)))
        < 10.0**-10.0
    ), "pvRvz calculated with R and z array input does not equal to calculated with scalar input"
    return None
