# Tests of the diskdf module: distribution functions from Dehnen (1999)
import os

import numpy
import pytest
from scipy import stats

from galpy.df import dehnendf, schwarzschilddf, shudf

_FEWERLONGINTEGRALS = True
# So we can reuse the following
ddf_correct_flat = None
ddf_correct2_flat = None
ddf_correct_powerrise = None
sdf_correct_flat = None


# First some tests of surfaceSigmaProfile and expSurfaceSigmaProfile
def test_expSurfaceSigmaProfile_surfacemass():
    from galpy.df import expSurfaceSigmaProfile

    essp = expSurfaceSigmaProfile(params=(0.25, 0.75, 0.1))
    assert (
        numpy.fabs(essp.surfacemass(0.5) - numpy.exp(-0.5 / 0.25)) < 10.0**-8.0
    ), "expSurfaceSigmaProfile's surfacemass does not work as expected"
    assert (
        numpy.fabs(essp.surfacemass(1.5, log=True) + 1.5 / 0.25) < 10.0**-8.0
    ), "expSurfaceSigmaProfile's surfacemass does not work as expected"
    return None


def test_expSurfaceSigmaProfile_surfacemassDerivative():
    from galpy.df import expSurfaceSigmaProfile

    essp = expSurfaceSigmaProfile(params=(0.25, 0.75, 0.1))
    assert (
        numpy.fabs(essp.surfacemassDerivative(0.5) + numpy.exp(-0.5 / 0.25) / 0.25)
        < 10.0**-8.0
    ), "expSurfaceSigmaProfile's surfacemassDerivative does not work as expected"
    assert (
        numpy.fabs(essp.surfacemassDerivative(1.5, log=True) + 1.0 / 0.25) < 10.0**-8.0
    ), "expSurfaceSigmaProfile's surfacemassDerivative does not work as expected"
    return None


def test_expSurfaceSigmaProfile_sigma2():
    from galpy.df import expSurfaceSigmaProfile

    essp = expSurfaceSigmaProfile(params=(0.25, 0.75, 0.1))
    assert (
        numpy.fabs(essp.sigma2(0.5) - 0.1**2.0 * numpy.exp(-(0.5 - 1.0) / 0.75 * 2.0))
        < 10.0**-8.0
    ), "expSurfaceSigmaProfile's sigma2 does not work as expected"
    assert (
        numpy.fabs(
            essp.sigma2(1.5, log=True) - 2.0 * numpy.log(0.1) + (1.5 - 1.0) / 0.75 * 2.0
        )
        < 10.0**-8.0
    ), "expSurfaceSigmaProfile's sigma2 does not work as expected"
    return None


def test_expSurfaceSigmaProfile_sigma2Derivative():
    from galpy.df import expSurfaceSigmaProfile

    essp = expSurfaceSigmaProfile(params=(0.25, 0.75, 0.1))
    assert (
        numpy.fabs(
            essp.sigma2Derivative(0.5)
            + 2.0 * 0.1**2.0 / 0.75 * numpy.exp(-(0.5 - 1.0) / 0.75 * 2.0)
        )
        < 10.0**-8.0
    ), "expSurfaceSigmaProfile's sigma2Derivative does not work as expected"
    assert (
        numpy.fabs(essp.sigma2Derivative(1.5, log=True) + 2.0 / 0.75) < 10.0**-8.0
    ), "expSurfaceSigmaProfile's sigma2 does not work as expected"
    return None


def test_surfaceSigmaProfile_outputParams():
    from galpy.df import expSurfaceSigmaProfile

    essp = expSurfaceSigmaProfile(params=(0.25, 0.75, 0.1))
    assert (
        numpy.fabs(essp.outputParams()[0] - 0.25) < 10.0**-8.0
    ), "surfaceSigmaProfile's outputParams does not behave as expected"
    assert (
        numpy.fabs(essp.outputParams()[1] - 0.75) < 10.0**-8.0
    ), "surfaceSigmaProfile's outputParams does not behave as expected"
    assert (
        numpy.fabs(essp.outputParams()[2] - 0.1) < 10.0**-8.0
    ), "surfaceSigmaProfile's outputParams does not behave as expected"
    return None


def test_surfaceSigmaProfile_formatStringParams():
    from galpy.df import expSurfaceSigmaProfile

    essp = expSurfaceSigmaProfile(params=(0.25, 0.75, 0.1))
    assert (
        essp.formatStringParams()[0] == r"%6.4f"
    ), "surfaceSigmaProfile's formatStringParams does not behave as expected"
    assert (
        essp.formatStringParams()[1] == r"%6.4f"
    ), "surfaceSigmaProfile's formatStringParams does not behave as expected"
    assert (
        essp.formatStringParams()[2] == r"%6.4f"
    ), "surfaceSigmaProfile's formatStringParams does not behave as expected"
    return None


def test_dfsetup_surfaceSigmaProfile():
    df = dehnendf(profileParams=(0.25, 0.75, 0.1), beta=0.0, correct=False)
    from galpy.df import expSurfaceSigmaProfile

    essp = expSurfaceSigmaProfile(params=(0.25, 0.75, 0.1))
    df_alt = dehnendf(surfaceSigma=essp, beta=0.0, correct=False)
    assert numpy.all(
        numpy.fabs(
            numpy.array(df._surfaceSigmaProfile._params)
            - numpy.array(df_alt._surfaceSigmaProfile._params)
        )
        < 10.0**-10.0
    ), "diskdf setup with explicit surfaceSigmaProfile class does not give the same profile as with parameters only"
    return None


# Tests for cold population, flat rotation curve: <vt> =~ v_c
def test_dehnendf_cold_flat_vt():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(df.meanvT(1.0) - 1.0) < 10.0**-3.0
    ), "mean vT of cold dehnendf in a flat rotation curve is not close to V_c at R=1"
    assert (
        numpy.fabs(df.meanvT(0.5) - 1.0) < 10.0**-3.0
    ), "mean vT of cold dehnendf in a flat rotation curve is not close to V_c at R=0.5"
    assert (
        numpy.fabs(df.meanvT(2.0) - 1.0) < 10.0**-3.0
    ), "mean vT of cold dehnendf in a flat rotation curve is not close to V_c at R=2"
    # Really close to the center
    assert (
        numpy.fabs(df.meanvT(0.0001) - 1.0) < 10.0**-3.0
    ), "mean vT of cold dehnendf in a flat rotation curve is not close to V_c at R=0.5"
    return None


# Tests for cold population, power-law rotation curve: <vt> =~ v_c
def test_dehnendf_cold_powerrise_vt():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    assert (
        numpy.fabs(df.meanvT(1.0) - 1.0) < 10.0**-3.0
    ), "mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=1"
    assert (
        numpy.fabs(df.meanvT(0.5) - (0.5) ** beta) < 10.0**-3.0
    ), "mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=0.5"
    assert (
        numpy.fabs(df.meanvT(2.0) - (2.0) ** beta) < 10.0**-3.0
    ), "mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=2"


def test_dehnendf_cold_powerfall_vt():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    assert (
        numpy.fabs(df.meanvT(1.0) - 1.0) < 10.0**-3.0
    ), "mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=1"
    assert (
        numpy.fabs(df.meanvT(0.5) - (0.5) ** beta) < 10.0**-3.0
    ), "mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=0.5"
    assert (
        numpy.fabs(df.meanvT(2.0) - (2.0) ** beta) < 10.0**-3.0
    ), "mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=2"
    return None


# Tests for cold population, flat rotation curve: <vt> =~ v_c
def test_dehnendf_cold_flat_skewvt():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvT(1.0)) < 1.0 / 20.0
        ), "skew vT of cold dehnendf in a flat rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.skewvT(0.5)) < 1.0 / 20.0
    ), "skew vT of cold dehnendf in a flat rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvT(2.0)) < 1.0 / 20.0
        ), "skew vT of cold dehnendf in a flat rotation curve is not close to zero at R=2"
    return None


# Tests for cold population, power-law rotation curve: <vt> =~ v_c
def test_dehnendf_cold_powerrise_skewvt():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvT(1.0)) < 1.0 / 20.0
        ), "skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.skewvT(0.5)) < 1.0 / 20.0
    ), "skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvT(2.0)) < 1.0 / 20.0
        ), "skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=2"
    return None


def test_dehnendf_cold_powerfall_skewvt():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvT(1.0)) < 1.0 / 20.0
        ), "skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.skewvT(0.5)) < 1.0 / 20.0
    ), "skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvT(2.0)) < 1.0 / 20.0
        ), "skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=2"
    return None


# Tests for cold population, flat rotation curve: <vr> = 0
def test_dehnendf_cold_flat_vr():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(df.meanvR(1.0) - 0.0) < 10.0**-3.0
    ), "mean vR of cold dehnendf in a flat rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.meanvR(0.5) - 0.0) < 10.0**-3.0
    ), "mean vR of cold dehnendf in a flat rotation curve is not close to zero at R=0.5"
    assert (
        numpy.fabs(df.meanvR(2.0) - 0.0) < 10.0**-3.0
    ), "mean vR of cold dehnendf in a flat rotation curve is not close to zero at R=2"
    return None


# Tests for cold population, flat rotation curve: kurtosis = 0
def test_dehnendf_cold_flat_kurtosisvt():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvT(1.0)) < 1.0 / 20.0
        ), "kurtosis vT of cold dehnendf in a flat rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.kurtosisvT(0.5)) < 1.0 / 20.0
    ), "kurtosis vT of cold dehnendf in a flat rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvT(2.0)) < 1.0 / 20.0
        ), "kurtosis vT of cold dehnendf in a flat rotation curve is not close to zero at R=2"
    return None


# Tests for cold population, power-law rotation curve: kurtosis = 0
def test_dehnendf_cold_powerrise_kurtosisvt():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvT(1.0)) < 1.0 / 20.0
        ), "kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.kurtosisvT(0.5)) < 1.0 / 20.0
    ), "kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvT(2.0)) < 1.0 / 20.0
        ), "kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=2"


def test_dehnendf_cold_powerfall_kurtosisvt():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvT(1.0)) < 1.0 / 20.0
        ), "kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.kurtosisvT(0.5)) < 1.0 / 20.0
    ), "kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvT(2.0)) < 1.0 / 20.0
        ), "kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=2"
    return None


# Tests for cold population, power-law rotation curve: <vr> = 0
def test_dehnendf_cold_powerrise_vr():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    assert (
        numpy.fabs(df.meanvR(1.0) - 0.0) < 10.0**-3.0
    ), "mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.meanvR(0.5) - 0.0) < 10.0**-3.0
    ), "mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5"
    assert (
        numpy.fabs(df.meanvR(2.0) - 0.0) < 10.0**-3.0
    ), "mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2"


def test_dehnendf_cold_powerfall_vr():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    assert (
        numpy.fabs(df.meanvR(1.0) - 0.0) < 10.0**-3.0
    ), "mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.meanvR(0.5) - 0.0) < 10.0**-3.0
    ), "mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5"
    assert (
        numpy.fabs(df.meanvR(2.0) - 0.0) < 10.0**-3.0
    ), "mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2"
    return None


# Tests for cold population, flat rotation curve: <vr> = 0
def test_dehnendf_cold_flat_skewvr():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvR(1.0) - 0.0) < 10.0**-3.0
        ), "skew vR of cold dehnendf in a flat rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.skewvR(0.5) - 0.0) < 10.0**-3.0
    ), "skew vR of cold dehnendf in a flat rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvR(2.0) - 0.0) < 10.0**-3.0
        ), "skew vR of cold dehnendf in a flat rotation curve is not close to zero at R=2"
    return None


# Tests for cold population, power-law rotation curve: <vr> = 0
def test_dehnendf_cold_powerrise_skewvr():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvR(1.0) - 0.0) < 10.0**-3.0
        ), "skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.skewvR(0.5) - 0.0) < 10.0**-3.0
    ), "skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvR(2.0) - 0.0) < 10.0**-3.0
        ), "skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2"


def test_dehnendf_cold_powerfall_skewvr():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvR(1.0) - 0.0) < 10.0**-3.0
        ), "skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.skewvR(0.5) - 0.0) < 10.0**-3.0
    ), "skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.skewvR(2.0) - 0.0) < 10.0**-3.0
        ), "skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2"
    return None


# Tests for cold population, flat rotation curve: kurtosis = 0
def test_dehnendf_cold_flat_kurtosisvr():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvR(1.0)) < 1.0 / 20.0
        ), "kurtosis vR of cold dehnendf in a flat rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.kurtosisvR(0.5)) < 1.0 / 20.0
    ), "kurtosis vR of cold dehnendf in a flat rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvR(2.0)) < 1.0 / 20.0
        ), "kurtosis vR of cold dehnendf in a flat rotation curve is not close to zero at R=2"
    return None


# Tests for cold population, power-law rotation curve: kurtosis = 0
def test_dehnendf_cold_powerrise_kurtosisvr():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvR(1.0)) < 1.0 / 20.0
        ), "kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.kurtosisvR(0.5)) < 1.0 / 20.0
    ), "kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvR(2.0)) < 1.0 / 20.0
        ), "kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2"


def test_dehnendf_cold_powerfall_kurtosisvr():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvR(1.0)) < 1.0 / 20.0
        ), "kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1"
    assert (
        numpy.fabs(df.kurtosisvR(0.5)) < 1.0 / 20.0
    ), "kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.kurtosisvR(2.0)) < 1.0 / 20.0
        ), "kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2"
    return None


# Tests for cold population, flat rotation curve: A = 0.5
def test_dehnendf_cold_flat_oortA():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(df.oortA(1.0) - 0.5 * 1.0 / 1.0) < 10.0**-3.0
    ), "Oort A of cold dehnendf in a flat rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortA(0.5) - 0.5 * 1.0 / 0.5) < 10.0**-3.0
    ), "Oort A of cold dehnendf in a flat rotation curve is not close to expected at R=0.5"
    # one w/ Romberg
    assert (
        numpy.fabs(df.oortA(2.0, romberg=True) - 0.5 * 1.0 / 2.0) < 10.0**-3.0
    ), "Oort A of cold dehnendf in a flat rotation curve is not close to expected at R=2"
    return None


# Tests for cold population, power-law rotation curve: A
def test_dehnendf_cold_powerrise_oortA():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortA(1.0) - 0.5 * 1.0 / 1.0 * (1.0 - beta)) < 10.0**-3.0
        ), "Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortA(0.5) - 0.5 * (0.5) ** beta / 0.5 * (1.0 - beta))
        < 10.0**-3.0
    ), "Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5"
    # one w/ Romberg
    assert (
        numpy.fabs(
            df.oortA(2.0, romberg=True) - 0.5 * (2.0) ** beta / 2.0 * (1.0 - beta)
        )
        < 10.0**-3.0
    ), "Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=2"
    return None


def test_dehnendf_cold_powerfall_oortA():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortA(1.0) - 0.5 * 1.0 / 1.0 * (1.0 - beta)) < 10.0**-3.0
        ), "Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortA(0.5) - 0.5 * (0.5) ** beta / 0.5 * (1.0 - beta))
        < 10.0**-3.0
    ), "Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5"
    # One w/ Romberg
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(
                df.oortA(2.0, romberg=True) - 0.5 * (2.0) ** beta / 2.0 * (1.0 - beta)
            )
            < 10.0**-3.0
        ), "Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=2"
    return None


# Tests for cold population, flat rotation curve: B = -0.5
def test_dehnendf_cold_flat_oortB():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(df.oortB(1.0) + 0.5 * 1.0 / 1.0) < 10.0**-3.0
    ), "Oort B of cold dehnendf in a flat rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortB(0.5) + 0.5 * 1.0 / 0.5) < 10.0**-3.0
    ), "Oort B of cold dehnendf in a flat rotation curve is not close to expected at R=0.5"
    assert (
        numpy.fabs(df.oortB(2.0) + 0.5 * 1.0 / 2.0) < 10.0**-3.0
    ), "Oort B of cold dehnendf in a flat rotation curve is not close to expected at R=2"
    return None


# Tests for cold population, power-law rotation curve: B
def test_dehnendf_cold_powerrise_oortB():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortB(1.0) + 0.5 * 1.0 / 1.0 * (1.0 + beta)) < 10.0**-3.0
        ), "Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortB(0.5) + 0.5 * (0.5) ** beta / 0.5 * (1.0 + beta))
        < 10.0**-3.0
    ), "Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortB(2.0) + 0.5 * (2.0) ** beta / 2.0 * (1.0 + beta))
            < 10.0**-3.0
        ), "Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=2"
    return None


def test_dehnendf_cold_powerfall_oortB():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortB(1.0) + 0.5 * 1.0 / 1.0 * (1.0 + beta)) < 10.0**-3.0
        ), "Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortB(0.5) + 0.5 * (0.5) ** beta / 0.5 * (1.0 + beta))
        < 10.0**-3.0
    ), "Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortB(2.0) + 0.5 * (2.0) ** beta / 2.0 * (1.0 + beta))
            < 10.0**-3.0
        ), "Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=2"
    return None


# Tests for cold population, flat rotation curve: C = 0
def test_dehnendf_cold_flat_oortC():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(df.oortC(1.0)) < 10.0**-3.0
    ), "Oort C of cold dehnendf in a flat rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortC(0.5)) < 10.0**-3.0
    ), "Oort C of cold dehnendf in a flat rotation curve is not close to expected at R=0.5"
    assert (
        numpy.fabs(df.oortC(2.0)) < 10.0**-3.0
    ), "Oort C of cold dehnendf in a flat rotation curve is not close to expected at R=2"
    return None


# Tests for cold population, power-law rotation curve: C
def test_dehnendf_cold_powerrise_oortC():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortC(1.0)) < 10.0**-3.0
        ), "Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortC(0.5)) < 10.0**-3.0
    ), "Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortC(2.0)) < 10.0**-3.0
        ), "Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=2"
    return None


def test_dehnendf_cold_powerfall_oortC():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortC(1.0)) < 10.0**-3.0
        ), "Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortC(0.5)) < 10.0**-3.0
    ), "Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortC(2.0)) < 10.0**-3.0
        ), "Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=2"
    return None


# Tests for cold population, flat rotation curve: K = 0
def test_dehnendf_cold_flat_oortK():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(df.oortK(1.0)) < 10.0**-3.0
    ), "Oort K of cold dehnendf in a flat rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortK(0.5)) < 10.0**-3.0
    ), "Oort K of cold dehnendf in a flat rotation curve is not close to expected at R=0.5"
    assert (
        numpy.fabs(df.oortK(2.0)) < 10.0**-3.0
    ), "Oort K of cold dehnendf in a flat rotation curve is not close to expected at R=2"
    return None


# Tests for cold population, power-law rotation curve: K
def test_dehnendf_cold_powerrise_oortK():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortK(1.0)) < 10.0**-3.0
        ), "Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortK(0.5)) < 10.0**-3.0
    ), "Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortK(2.0)) < 10.0**-3.0
        ), "Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=2"
    return None


def test_dehnendf_cold_powerfall_oortK():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortK(1.0)) < 10.0**-3.0
        ), "Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.oortK(0.5)) < 10.0**-3.0
    ), "Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5"
    if not _FEWERLONGINTEGRALS:
        assert (
            numpy.fabs(df.oortK(2.0)) < 10.0**-3.0
        ), "Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=2"
    return None


# Tests for cold population, flat rotation curve: sigma_R^2 / sigma_T^2 = kappa^2 / Omega^2
def test_dehnendf_cold_flat_srst():
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(df.sigmaR2(1.0) / df.sigmaT2(1.0) - 2.0) < 10.0**-2.0
    ), "sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.sigmaR2(0.5) / df.sigmaT2(0.5) - 2.0) < 10.0**-2.0
    ), "sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.sigmaR2(2.0) / df.sigmaT2(2.0) - 2.0) < 10.0**-2.0
    ), "sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1"
    return None


# Tests for cold population, power-law rotation curve: sigma_R^2 / sigma_T^2 = kappa^2 / Omega^2
def test_dehnendf_cold_powerrise_srst():
    # Rising rotation curve
    beta = 0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    assert (
        numpy.fabs(df.sigmaR2(1.0) / df.sigmaT2(1.0) - 2.0 / (1.0 + beta)) < 10.0**-2.0
    ), "sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.sigmaR2(0.5) / df.sigmaT2(0.5) - 2.0 / (1.0 + beta)) < 10.0**-2.0
    ), "sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.sigmaR2(2.0) / df.sigmaT2(2.0) - 2.0 / (1.0 + beta)) < 10.0**-2.0
    ), "sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1"
    return None


def test_dehnendf_cold_powerfall_srst():
    # Falling rotation curve
    beta = -0.2
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=beta, correct=False
    )
    assert (
        numpy.fabs(df.sigmaR2(1.0) / df.sigmaT2(1.0) - 2.0 / (1.0 + beta)) < 10.0**-2.0
    ), "sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.sigmaR2(0.5) / df.sigmaT2(0.5) - 2.0 / (1.0 + beta)) < 10.0**-2.0
    ), "sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1"
    assert (
        numpy.fabs(df.sigmaR2(2.0) / df.sigmaT2(2.0) - 2.0 / (1.0 + beta)) < 10.0**-2.0
    ), "sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1"
    return None


def test_targetSigma2():
    beta = 0.0
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.1), beta=beta, correct=False
    )
    assert (
        numpy.fabs(df.targetSigma2(1.0) - 0.1**2.0) < 10.0**-8.0
    ), "targetSigma2 for dehnendf does not agree with input"
    assert (
        numpy.fabs(df.targetSigma2(0.3) - 0.1**2.0 * numpy.exp(-(0.3 - 1.0) / 0.5))
        < 10.0**-8.0
    ), "targetSigma2 for dehnendf does not agree with input"
    assert (
        numpy.fabs(
            df.targetSigma2(3.0, log=True) - numpy.log(0.1) * 2.0 + (3.0 - 1.0) / 0.5
        )
        < 10.0**-8.0
    ), "targetSigma2 for dehnendf does not agree with input"
    return None


def test_targetSurfacemass():
    beta = 0.0
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.1), beta=beta, correct=False
    )
    assert (
        numpy.fabs(df.targetSurfacemass(1.0) - numpy.exp(-1.0 / 0.3333333333333333))
        < 10.0**-8.0
    ), "targetSigma2 for dehnendf does not agree with input"
    assert (
        numpy.fabs(df.targetSurfacemass(0.3) - numpy.exp(-0.3 / 0.3333333333333333))
        < 10.0**-8.0
    ), "targetSigma2 for dehnendf does not agree with input"
    assert (
        numpy.fabs(df.targetSurfacemass(3.0, log=True) + 3.0 / 0.3333333333333333)
        < 10.0**-8.0
    ), "targetSigma2 for dehnendf does not agree with input"
    return None


def test_targetSurfacemassLOS():
    beta = 0.0
    df = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.1), beta=beta, correct=False
    )
    # Some easy directions in l
    assert (
        numpy.fabs(
            df.targetSurfacemassLOS(0.2, l=0.0, deg=True)
            - 0.2 * numpy.exp(-0.8 / 0.3333333333333333)
        )
        < 10.0**-8.0
    ), "targetSigma2 for dehnendf does not agree with input"
    assert (
        numpy.fabs(
            df.targetSurfacemassLOS(0.2, l=180.0, deg=True)
            - 0.2 * numpy.exp(-1.2 / 0.3333333333333333)
        )
        < 10.0**-8.0
    ), "targetSigma2 for dehnendf does not agree with input"
    assert (
        numpy.fabs(
            df.targetSurfacemassLOS(0.2, l=numpy.pi, deg=False)
            - 0.2 * numpy.exp(-1.2 / 0.3333333333333333)
        )
        < 10.0**-8.0
    ), "targetSigma2 for dehnendf does not agree with input"
    assert (
        numpy.fabs(
            df.targetSurfacemassLOS(0.2, l=numpy.pi / 2.0, log=True, deg=False)
            - numpy.log(0.2)
            + numpy.sqrt(1.0 + 0.2**2.0 - 2.0 * 0.2 * numpy.cos(numpy.pi / 2.0))
            / 0.3333333333333333
        )
        < 10.0**-8.0
    ), "targetSigma2 for dehnendf does not agree with input"
    return None


def test_cold_surfacemass():
    dfc = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(
            numpy.log(dfc.surfacemass(0.9)) - numpy.log(dfc.targetSurfacemass(0.9))
        )
        < 0.01
    ), "True surfacemass deviates more from target surfacemass for cold Dehnen DF than expected"
    assert (
        numpy.fabs(
            numpy.log(dfc.surfacemass(0.5)) - numpy.log(dfc.targetSurfacemass(0.5))
        )
        < 0.01
    ), "True surfacemass deviates more from target surfacemass for cold Dehnen DF than expected"
    assert (
        numpy.fabs(
            numpy.log(dfc.surfacemass(2.0)) - numpy.log(dfc.targetSurfacemass(2.0))
        )
        < 0.01
    ), "True surfacemass deviates more from target surfacemass for cold Dehnen DF than expected"
    return None


def test_surfacemass():
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    assert (
        numpy.fabs(
            numpy.log(dfc.surfacemass(0.9)) - numpy.log(dfc.targetSurfacemass(0.9))
        )
        < 0.05
    ), "True surfacemass deviates more from target surfacemass for Dehnen DF with documentation-example parameters than expected"
    assert (
        numpy.fabs(
            numpy.log(dfc.surfacemass(0.05)) - numpy.log(dfc.targetSurfacemass(0.05))
        )
        < 0.5
    ), "True surfacemass deviates more from target surfacemass for Dehnen DF with documentation-example parameters than expected"
    assert (
        numpy.fabs(numpy.log(dfc.surfacemass(4.0, romberg=True, relative=True))) < 0.05
    ), "True surfacemass deviates more from target surfacemass for Dehnen DF with documentation-example parameters than expected"
    return None


def test_cold_sigma2surfacemass():
    dfc = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(
            numpy.log(dfc.sigma2surfacemass(0.9))
            - numpy.log(dfc.targetSigma2(0.9) * dfc.targetSurfacemass(0.9))
        )
        < 0.01
    ), "True surfacemass deviates more from target surfacemass for cold Dehnen DF than expected"
    assert (
        numpy.fabs(
            numpy.log(dfc.sigma2surfacemass(0.5))
            - numpy.log(dfc.targetSigma2(0.5) * dfc.targetSurfacemass(0.5))
        )
        < 0.01
    ), "True surfacemass deviates more from target surfacemass for cold Dehnen DF than expected"
    assert (
        numpy.fabs(
            numpy.log(dfc.sigma2surfacemass(2.0))
            - numpy.log(dfc.targetSigma2(2.0) * dfc.targetSurfacemass(2.0))
        )
        < 0.01
    ), "True surfacemass deviates more from target surfacemass for cold Dehnen DF than expected"
    return None


def test_sigma2surfacemass():
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    assert (
        numpy.fabs(
            numpy.log(dfc.sigma2surfacemass(0.9))
            - numpy.log(dfc.targetSigma2(0.9) * dfc.targetSurfacemass(0.9))
        )
        < 0.05
    ), "True surfacemass deviates more from target surfacemass for Dehnen DF with documentation-example parameters than expected"
    assert (
        numpy.fabs(
            numpy.log(dfc.sigma2surfacemass(0.3))
            - numpy.log(dfc.targetSigma2(0.3) * dfc.targetSurfacemass(0.3))
        )
        < 0.2
    ), "True surfacemass deviates more from target surfacemass for Dehnen DF with documentation-example parameters than expected"
    assert (
        numpy.fabs(numpy.log(dfc.sigma2surfacemass(3.0, relative=True, romberg=True)))
        < 0.1
    ), "True surfacemass deviates more from target surfacemass for Dehnen DF with documentation-example parameters than expected"
    return None


def test_vmomentsurfacemass():
    # Test that vmomentsurfacemass gives reasonable results
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.4))
    assert (
        numpy.fabs(dfc.vmomentsurfacemass(0.9, 0.0, 0.0) - dfc.surfacemass(0.9))
        < 10.0**-8.0
    ), "vmomentsurfacemass with (n,m) = (0,0) is not equal to surfacemass"
    assert (
        numpy.fabs(
            dfc.vmomentsurfacemass(0.9, 0.0, 0.0, relative=True)
            - dfc.surfacemass(0.9) / dfc.targetSurfacemass(0.9)
        )
        < 10.0**-8.0
    ), "vmomentsurfacemass with (n,m) = (0,0) and relative=True is not equal to surfacemass/targetSurfacemass"
    assert (
        numpy.fabs(dfc.vmomentsurfacemass(0.9, 2.0, 0.0) - dfc.sigma2surfacemass(0.9))
        < 10.0**-8.0
    ), "vmomentsurfacemass with (n,m) = (2,0) is not equal to sigma2surfacemass"
    assert (
        numpy.fabs(dfc.vmomentsurfacemass(0.9, 1.0, 1.0, romberg=True)) < 10.0**-8.0
    ), "vmomentsurfacemass with (n,m) = (1.,1.) is not equal to zero (not automatically zero)"
    assert (
        numpy.fabs(dfc.vmomentsurfacemass(0.9, 1, 1)) < 10.0**-8.0
    ), "vmomentsurfacemass with (n,m) = (1,1) is not equal to zero"
    return None


def test_vmomentsurfacemass_physical():
    # Test that vmomentsurfacemass gives correct physical results
    from galpy.util import conversion

    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    ro, vo = 7.0, 230.0
    assert (
        numpy.fabs(
            dfc.vmomentsurfacemass(0.9, 0.0, 0.0, use_physical=True, ro=ro, vo=vo)
            - dfc.vmomentsurfacemass(0.9, 0.0, 0.0)
            * conversion.surfdens_in_msolpc2(vo, ro)
        )
        < 10.0**-8.0
    ), "vmomentsurfacemass with (n,m) = (0,0) is not equal to surfacemass"
    assert (
        numpy.fabs(
            dfc.vmomentsurfacemass(0.9, 1.0, 0.0, use_physical=True, ro=ro, vo=vo)
            - dfc.vmomentsurfacemass(0.9, 1.0, 0.0)
            * vo
            * conversion.surfdens_in_msolpc2(vo, ro)
        )
        < 10.0**-8.0
    ), "vmomentsurfacemass with (n,m) = (0,0) is not equal to surfacemass"
    assert (
        numpy.fabs(
            dfc.vmomentsurfacemass(0.9, 1.0, 2.0, use_physical=True, ro=ro, vo=vo)
            - dfc.vmomentsurfacemass(0.9, 1.0, 2.0)
            * vo**3.0
            * conversion.surfdens_in_msolpc2(vo, ro)
        )
        < 10.0**-8.0
    ), "vmomentsurfacemass with (n,m) = (0,0) is not equal to surfacemass"
    return None


def test_cold_surfacemassLOS():
    dfc = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.01), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(
            numpy.log(dfc.surfacemassLOS(0.1, 0.0, target=False))
            - numpy.log(0.1 * dfc.targetSurfacemass(0.9))
        )
        < 0.01
    ), "True surfacemassLOS deviates more from target surfacemassLOS for cold Dehnen DF than expected"
    assert (
        numpy.fabs(
            numpy.log(
                dfc.surfacemassLOS(
                    numpy.cos(numpy.pi / 6.0), numpy.pi / 6.0, target=False, deg=False
                )
            )
            - numpy.log(
                numpy.cos(numpy.pi / 6.0)
                * dfc.targetSurfacemass(numpy.cos(numpy.pi / 3.0))
            )
        )
        < 0.01
    ), "True surfacemassLOS deviates more from target surfacemassLOS for cold Dehnen DF than expected"
    assert (
        numpy.fabs(
            numpy.log(
                dfc.surfacemassLOS(
                    numpy.cos(numpy.pi / 3.0), numpy.pi / 3.0, deg=False, target=True
                )
            )
            - numpy.log(
                numpy.cos(numpy.pi / 3.0)
                * dfc.targetSurfacemass(numpy.cos(numpy.pi / 6.0))
            )
        )
        < 0.01
    ), "True surfacemassLOS deviates more from target surfacemassLOS for cold Dehnen DF than expected"
    assert (
        numpy.fabs(
            numpy.log(
                dfc.surfacemassLOS(
                    numpy.cos(numpy.pi / 3.0),
                    numpy.pi / 3.0,
                    deg=False,
                    relative=True,
                    target=True,
                )
            )
            - numpy.log(numpy.cos(numpy.pi / 3.0))
        )
        < 0.01
    ), "True surfacemassLOS deviates more from target surfacemassLOS for cold Dehnen DF than expected"
    return None


def test_warm_surfacemassLOS():
    dfc = dehnendf(
        profileParams=(0.3333333333333333, 1.0, 0.1), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(
            numpy.log(dfc.surfacemassLOS(0.1, 0.0, target=False))
            - numpy.log(0.1 * dfc.surfacemass(0.9))
        )
        < 10.0**-6.0
    ), "surfacemassLOS deviates more from surfacemass for warm Dehnen DF than expected"
    assert (
        numpy.fabs(
            numpy.log(
                dfc.surfacemassLOS(
                    numpy.cos(numpy.pi / 6.0), numpy.pi / 6.0, target=False, deg=False
                )
            )
            - numpy.log(
                numpy.cos(numpy.pi / 6.0) * dfc.surfacemass(numpy.cos(numpy.pi / 3.0))
            )
        )
        < 0.01
    ), "surfacemassLOS deviates more from target surfacemass for warm Dehnen DF than expected"
    assert (
        numpy.fabs(
            numpy.log(
                dfc.surfacemassLOS(
                    numpy.cos(numpy.pi / 3.0), numpy.pi / 3.0, deg=False, target=True
                )
            )
            - numpy.log(
                numpy.cos(numpy.pi / 3.0)
                * dfc.targetSurfacemass(numpy.cos(numpy.pi / 6.0))
            )
        )
        < 0.01
    ), "surfacemassLOS w/ target deviates more from target surfacemassLOS for warm Dehnen DF than expected"
    assert (
        numpy.fabs(
            numpy.log(
                dfc.surfacemassLOS(
                    numpy.cos(numpy.pi / 3.0),
                    numpy.pi / 3.0,
                    deg=False,
                    relative=True,
                    target=True,
                )
            )
            - numpy.log(numpy.cos(numpy.pi / 3.0))
        )
        < 0.01
    ), "surfacemassLOS w/ target deviates more from target surfacemass for warm Dehnen DF than expected"
    return None


def test_dehnendf_call_sanity():
    # Sanity checking of dehnendf's call function
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    meanvt = dfc.meanvT(0.7)
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.0, meanvt / 2.0])
    ), "dehnendf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.0, meanvt * 2.0])
    ), "dehnendf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, -0.1, meanvt])
    ), "dehnendf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.1, meanvt])
    ), "dehnendf does not peak near (vR,vT) = (0,meanvT)"
    return None


def test_shudf_call_sanity_flat():
    # Sanity checking of shudf's call function
    dfc = shudf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    meanvt = dfc.meanvT(0.7)
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.0, meanvt / 2.0])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.0, meanvt * 2.0])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, -0.1, meanvt])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.1, meanvt])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    assert (
        dfc(numpy.array([0.7, 0.0, -0.1])) == 0.0
    ), "shudf not zero for counter-rotating orbits"
    return None


def test_shudf_call_sanity_powerfall():
    # Sanity checking of shudf's call function
    dfc = shudf(beta=-0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    meanvt = dfc.meanvT(0.7)
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.0, meanvt / 2.0])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.0, meanvt * 2.0])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, -0.1, meanvt])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.1, meanvt])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    return None


def test_shudf_call_sanity_powerrise():
    # Sanity checking of shudf's call function
    dfc = shudf(beta=0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    meanvt = dfc.meanvT(0.7, nsigma=3.0)
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.0, meanvt / 2.0])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.0, meanvt * 2.0])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, -0.1, meanvt])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    assert dfc(numpy.array([0.7, 0.0, meanvt])) > dfc(
        numpy.array([0.7, 0.1, meanvt])
    ), "shudf does not peak near (vR,vT) = (0,meanvT)"
    return None


def test_call_diffinputs():
    from galpy.orbit import Orbit

    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    R, vR, vT, phi = 0.8, 0.4, 1.1, 2.0
    to = Orbit([R, vR, vT, phi])
    tao = Orbit([R, vR, vT])
    # R,vR,vT,phi vs R,vR,vT
    assert (
        numpy.fabs(dfc(numpy.array([R, vR, vT, phi])) - dfc(numpy.array([R, vR, vT])))
        < 10.0**-10.0
    ), "diskdf __call__ w/ array R,vR,vT,phi neq w/ array R,vR,vT"
    # orbit vs R,vR,vT
    assert (
        numpy.fabs(dfc(to) - dfc(numpy.array([R, vR, vT]))) < 10.0**-10.0
    ), "diskdf __call__ w/ orbit neq w/ array R,vR,vT"
    # axi orbit vs R,vR,vT
    assert (
        numpy.fabs(dfc(tao) - dfc(numpy.array([R, vR, vT]))) < 10.0**-10.0
    ), "diskdf __call__ w/ axi orbit neq w/ array R,vR,vT"
    # orbit w/ t vs R,vR,vT
    assert (
        numpy.fabs(dfc(to, 0.0) - dfc(numpy.array([R, vR, vT]))) < 10.0**-10.0
    ), "diskdf __call__ w/ orbit and t neq w/ array R,vR,vT"
    # axi orbit w/ t vs R,vR,vT
    assert (
        numpy.fabs(dfc(tao, 0.0) - dfc(numpy.array([R, vR, vT]))) < 10.0**-10.0
    ), "diskdf __call__ w/ axi orbit and t neq w/ array R,vR,vT"
    # list of orbit vs R,vR,vT
    assert (
        numpy.fabs(dfc([to]) - dfc(numpy.array([R, vR, vT]))) < 10.0**-10.0
    ), "diskdf __call__ w/ list of orbit neq w/ array R,vR,vT"
    # E,L vs R,vR,vT
    assert (
        numpy.fabs(
            dfc(numpy.log(R) + vR**2.0 / 2.0 + vT**2.0 / 2.0, R * vT)
            - dfc(numpy.array([R, vR, vT]))
        )
        < 10.0**-10.0
    ), "diskdf __call__ w/ E,L and t neq w/ array R,vR,vT"
    return None


def test_call_marginalizevperp():
    from galpy.orbit import Orbit

    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.4))
    # l=0
    R, vR = 1.8, 0.4
    vts = numpy.linspace(0.0, 1.5, 51)
    pvts = numpy.array([dfc(numpy.array([R, vR, vt])) for vt in vts])
    assert (
        numpy.fabs(
            numpy.sum(pvts) * (vts[1] - vts[0])
            - dfc(Orbit([R, vR, 0.0, 0.0]), marginalizeVperp=True)
        )
        < 10.0**-4.0
    ), "diskdf call w/ marginalizeVperp does not work"
    # Another l=0, where va > sigmaR1
    R, vR = 1.25, 0.4
    vts = numpy.linspace(0.0, 1.5, 51)
    pvts = numpy.array([dfc(numpy.array([R, vR, vt])) for vt in vts])
    assert (
        numpy.fabs(
            numpy.sum(pvts) * (vts[1] - vts[0])
            - dfc(Orbit([R, vR, 0.0, 0.0]), marginalizeVperp=True)
        )
        < 10.0**-4.0
    ), "diskdf call w/ marginalizeVperp does not work"
    # l=270
    R, vT = numpy.sin(numpy.pi / 6.0), 0.7  # l=30 degree
    vrs = numpy.linspace(-2.0, 2.0, 101)
    pvrs = numpy.array([dfc(numpy.array([R, vr, vT])) for vr in vrs])
    assert (
        numpy.fabs(
            numpy.sum(pvrs) * (vrs[1] - vrs[0])
            - dfc(Orbit([R, 0.0, vT, -numpy.pi / 3.0]), marginalizeVperp=True, nsigma=4)
        )
        < 10.0**-4.0
    ), "diskdf call w/ marginalizeVperp does not work"
    return None


def test_call_marginalizevlos():
    from galpy.orbit import Orbit

    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.4))
    # l=0
    R, vT = 0.8, 0.7
    vrs = numpy.linspace(-2.0, 2.0, 101)
    pvrs = numpy.array([dfc(numpy.array([R, vr, vT])) for vr in vrs])
    assert (
        numpy.fabs(
            numpy.sum(pvrs) * (vrs[1] - vrs[0])
            - dfc(Orbit([R, 0.0, vT, 0.0]), marginalizeVlos=True)
        )
        < 10.0**-4.0
    ), "diskdf call w/ marginalizeVlos does not work"
    # l=270
    R, vR = numpy.sin(numpy.pi / 6.0), 0.4  # l=30 degree
    vts = numpy.linspace(-2.5, 2.5, 101)
    pvts = numpy.array([dfc(numpy.array([R, vR, vt])) for vt in vts])
    assert (
        numpy.fabs(
            numpy.sum(pvts) * (vts[1] - vts[0])
            - dfc(Orbit([R, vR, 0.0, -numpy.pi / 3.0]), marginalizeVlos=True, nsigma=4)
        )
        < 10.0**-4.0
    ), "diskdf call w/ marginalizeVlos does not work"
    return None


def test_dehnendf_dlnfdR_flat():
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdR(R, vR, vT)) < 10.0**-6.0
    ), "dehnendf's dlnfdR does not work"
    return None


def test_dehnendf_dlnfdR_powerfall():
    dfc = dehnendf(beta=-0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dR = 10**-6.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdR(R, vR, vT)) < 10.0**-6.0
    ), "dehnendf's dlnfdR does not work"
    return None


def test_dehnendf_dlnfdR_powerrise():
    dfc = dehnendf(beta=0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdR(R, vR, vT)) < 10.0**-6.0
    ), "dehnendf's dlnfdR does not work"
    return None


def test_dehnendf_dlnfdvR_flat():
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvR(R, vR, vT)) < 10.0**-6.0
    ), "dehnendf's dlnfdvR does not work"
    return None


def test_dehnendf_dlnfdvR_powerfall():
    dfc = dehnendf(beta=-0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvR(R, vR, vT)) < 10.0**-6.0
    ), "dehnendf's dlnfdvR does not work"
    return None


def test_dehnendf_dlnfdvR_powerrise():
    dfc = dehnendf(beta=0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvR(R, vR, vT)) < 10.0**-6.0
    ), "dehnendf's dlnfdvR does not work"
    return None


def test_dehnendf_dlnfdvT_flat():
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvT = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvT(R, vR, vT)) < 10.0**-6.0
    ), "dehnendf's dlnfdvT does not work"
    return None


def test_dehnendf_dlnfdvT_powerfall():
    dfc = dehnendf(beta=-0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvT = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvT(R, vR, vT)) < 10.0**-6.0
    ), "dehnendf's dlnfdvT does not work"
    return None


def test_dehnendf_dlnfdvT_powerrise():
    dfc = dehnendf(beta=0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvT = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvT(R, vR, vT)) < 10.0**-6.0
    ), "dehnendf's dlnfdvT does not work"
    return None


def test_dehnendf_dlnfdRe_flat():
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Calculate dlndfdRe w/ the chain rule; first calculate dR
    dR = 10**-6.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnfdR = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + numpy.log(R)
    RE = numpy.exp(E - 0.5)
    dE = vR**2.0 / 2.0 + vT**2.0 / 2.0 + numpy.log(R + dR)
    dRE = numpy.exp(dE - 0.5)
    dRedR = (dRE - RE) / dR
    # dvR
    dvR = 10**-6.0
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnfdvR = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    dE = (vR + dvR) ** 2.0 / 2.0 + vT**2.0 / 2.0 + numpy.log(R)
    dRE = numpy.exp(dE - 0.5)
    dRedvR = (dRE - RE) / dvR
    # dvT
    dvT = 10**-6.0
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnfdvT = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    dE = vR**2.0 / 2.0 + (vT + dvT) ** 2.0 / 2.0 + numpy.log(R)
    dRE = numpy.exp(dE - 0.5)
    dRedvT = (dRE - RE) / dvT
    # Calculate dR/dRe etc. from matrix inversion
    dRvRvTdRe = numpy.linalg.inv(
        numpy.array([[dRedR, dRedvR, dRedvT], [vT, 0.0, R], [0.0, 1.0, 0.0]])
    )
    dlnf = (
        dlnfdR * dRvRvTdRe[0, 0] + dlnfdvR * dRvRvTdRe[1, 0] + dlnfdvT * dRvRvTdRe[2, 0]
    )
    assert (
        numpy.fabs(dlnf - dfc._dlnfdRe(R, vR, vT)) < 10.0**-5.0
    ), "dehnendf's dlnfdRe does not work"
    return None


def test_dehnendf_dlnfdRe_powerfall():
    beta = -0.2
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Calculate dlndfdRe w/ the chain rule; first calculate dR
    dR = 10**-6.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnfdR = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    RE = (2.0 * E / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dE = vR**2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * (R + dR) ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedR = (dRE - RE) / dR
    # dvR
    dvR = 10**-6.0
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnfdvR = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    dE = (vR + dvR) ** 2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedvR = (dRE - RE) / dvR
    # dvT
    dvT = 10**-6.0
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnfdvT = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    dE = vR**2.0 / 2.0 + (vT + dvT) ** 2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedvT = (dRE - RE) / dvT
    # Calculate dR/dRe etc. from matrix inversion
    dRvRvTdRe = numpy.linalg.inv(
        numpy.array([[dRedR, dRedvR, dRedvT], [vT, 0.0, R], [0.0, 1.0, 0.0]])
    )
    dlnf = (
        dlnfdR * dRvRvTdRe[0, 0] + dlnfdvR * dRvRvTdRe[1, 0] + dlnfdvT * dRvRvTdRe[2, 0]
    )
    assert (
        numpy.fabs(dlnf - dfc._dlnfdRe(R, vR, vT)) < 10.0**-5.0
    ), "dehnendf's dlnfdRe does not work"
    return None


def test_dehnendf_dlnfdRe_powerrise():
    beta = 0.2
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Calculate dlndfdRe w/ the chain rule; first calculate dR
    dR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnfdR = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    RE = (2.0 * E / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dE = vR**2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * (R + dR) ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedR = (dRE - RE) / dR
    # dvR
    dvR = 10**-8.0
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnfdvR = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    dE = (vR + dvR) ** 2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedvR = (dRE - RE) / dvR
    # dvT
    dvT = 10**-8.0
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnfdvT = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    dE = vR**2.0 / 2.0 + (vT + dvT) ** 2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedvT = (dRE - RE) / dvT
    # Calculate dR/dRe etc. from matrix inversion
    dRvRvTdRe = numpy.linalg.inv(
        numpy.array([[dRedR, dRedvR, dRedvT], [vT, 0.0, R], [0.0, 1.0, 0.0]])
    )
    dlnf = (
        dlnfdR * dRvRvTdRe[0, 0] + dlnfdvR * dRvRvTdRe[1, 0] + dlnfdvT * dRvRvTdRe[2, 0]
    )
    assert (
        numpy.fabs(dlnf - dfc._dlnfdRe(R, vR, vT)) < 10.0**-5.0
    ), "dehnendf's dlnfdRe does not work"
    return None


def test_dehnendf_dlnfdl_flat():
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Calculate dlndfdl w/ the chain rule; first calculate dR
    dR = 10**-6.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnfdR = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + numpy.log(R)
    RE = numpy.exp(E - 0.5)
    dE = vR**2.0 / 2.0 + vT**2.0 / 2.0 + numpy.log(R + dR)
    dRE = numpy.exp(dE - 0.5)
    dRedR = (dRE - RE) / dR
    # dvR
    dvR = 10**-6.0
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnfdvR = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    dE = (vR + dvR) ** 2.0 / 2.0 + vT**2.0 / 2.0 + numpy.log(R)
    dRE = numpy.exp(dE - 0.5)
    dRedvR = (dRE - RE) / dvR
    # dvT
    dvT = 10**-6.0
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnfdvT = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    dE = vR**2.0 / 2.0 + (vT + dvT) ** 2.0 / 2.0 + numpy.log(R)
    dRE = numpy.exp(dE - 0.5)
    dRedvT = (dRE - RE) / dvT
    # Calculate dR/dl etc. from matrix inversion
    dRvRvTdl = numpy.linalg.inv(
        numpy.array([[dRedR, dRedvR, dRedvT], [vT, 0.0, R], [0.0, 1.0, 0.0]])
    )
    dlnf = dlnfdR * dRvRvTdl[0, 1] + dlnfdvR * dRvRvTdl[1, 1] + dlnfdvT * dRvRvTdl[2, 1]
    assert (
        numpy.fabs(dlnf - dfc._dlnfdl(R, vR, vT)) < 10.0**-5.0
    ), "dehnendf's dlnfdl does not work"
    return None


def test_dehnendf_dlnfdl_powerfall():
    beta = -0.2
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Calculate dlndfdl w/ the chain rule; first calculate dR
    dR = 10**-6.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnfdR = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    RE = (2.0 * E / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dE = vR**2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * (R + dR) ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedR = (dRE - RE) / dR
    # dvR
    dvR = 10**-6.0
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnfdvR = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    dE = (vR + dvR) ** 2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedvR = (dRE - RE) / dvR
    # dvT
    dvT = 10**-6.0
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnfdvT = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    dE = vR**2.0 / 2.0 + (vT + dvT) ** 2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedvT = (dRE - RE) / dvT
    # Calculate dR/dl etc. from matrix inversion
    dRvRvTdl = numpy.linalg.inv(
        numpy.array([[dRedR, dRedvR, dRedvT], [vT, 0.0, R], [0.0, 1.0, 0.0]])
    )
    dlnf = dlnfdR * dRvRvTdl[0, 1] + dlnfdvR * dRvRvTdl[1, 1] + dlnfdvT * dRvRvTdl[2, 1]
    assert (
        numpy.fabs(dlnf - dfc._dlnfdl(R, vR, vT)) < 10.0**-5.0
    ), "dehnendf's dlnfdl does not work"
    return None


def test_dehnendf_dlnfdl_powerrise():
    beta = 0.2
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Calculate dlndfdl w/ the chain rule; first calculate dR
    dR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnfdR = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    E = vR**2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    RE = (2.0 * E / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dE = vR**2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * (R + dR) ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedR = (dRE - RE) / dR
    # dvR
    dvR = 10**-8.0
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnfdvR = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    dE = (vR + dvR) ** 2.0 / 2.0 + vT**2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedvR = (dRE - RE) / dvR
    # dvT
    dvT = 10**-8.0
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnfdvT = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    dE = vR**2.0 / 2.0 + (vT + dvT) ** 2.0 / 2.0 + 1.0 / 2.0 / beta * R ** (2.0 * beta)
    dRE = (2.0 * dE / (1.0 + 1.0 / beta)) ** (1.0 / 2.0 / beta)
    dRedvT = (dRE - RE) / dvT
    # Calculate dR/dl etc. from matrix inversion
    dRvRvTdl = numpy.linalg.inv(
        numpy.array([[dRedR, dRedvR, dRedvT], [vT, 0.0, R], [0.0, 1.0, 0.0]])
    )
    dlnf = dlnfdR * dRvRvTdl[0, 1] + dlnfdvR * dRvRvTdl[1, 1] + dlnfdvT * dRvRvTdl[2, 1]
    assert (
        numpy.fabs(dlnf - dfc._dlnfdl(R, vR, vT)) < 10.0**-5.0
    ), "dehnendf's dlnfdl does not work"
    return None


def test_shudf_dlnfdR_flat():
    dfc = shudf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdR(R, vR, vT)) < 10.0**-6.0
    ), "shudf's dlnfdR does not work"
    return None


def test_shudf_dlnfdR_powerfall():
    dfc = shudf(beta=-0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dR = 10**-6.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    # print((dfc._dlnfdR(R+dR,vR,vT)-dfc._dlnfdR(R,vR,vT))/dR)
    dlnf = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdR(R, vR, vT)) < 10.0**-6.0
    ), "shudf's dlnfdR does not work"
    return None


def test_shudf_dlnfdR_powerrise():
    dfc = shudf(beta=0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    Rn = R + dR
    dR = Rn - R  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R + dR, vR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdR(R, vR, vT)) < 10.0**-6.0
    ), "shudf's dlnfdR does not work"
    return None


def test_shudf_dlnfdvR_flat():
    dfc = shudf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvR(R, vR, vT)) < 10.0**-6.0
    ), "shudf's dlnfdvR does not work"
    return None


def test_shudf_dlnfdvR_powerfall():
    dfc = shudf(beta=-0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvR(R, vR, vT)) < 10.0**-6.0
    ), "shudf's dlnfdvR does not work"
    return None


def test_shudf_dlnfdvR_powerrise():
    dfc = shudf(beta=0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvR = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vRn = vR + dvR
    dvR = vRn - vR  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR + dvR, vT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvR
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvR(R, vR, vT)) < 10.0**-6.0
    ), "shudf's dlnfdvR does not work"
    return None


def test_shudf_dlnfdvT_flat():
    dfc = shudf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvT = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvT(R, vR, vT)) < 10.0**-6.0
    ), "shudf's dlnfdvT does not work"
    return None


def test_shudf_dlnfdvT_powerfall():
    dfc = shudf(beta=-0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvT = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvT(R, vR, vT)) < 10.0**-6.0
    ), "shudf's dlnfdvT does not work"
    return None


def test_shudf_dlnfdvT_powerrise():
    dfc = shudf(beta=0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
    dvT = 10**-8.0
    R, vR, vT = 0.8, 0.1, 0.9
    vTn = vT + dvT
    dvT = vTn - vT  # representable number
    dlnf = (
        numpy.log(dfc(numpy.array([R, vR, vT + dvT])))
        - numpy.log(dfc(numpy.array([R, vR, vT])))
    ) / dvT
    assert (
        numpy.fabs(dlnf - dfc._dlnfdvT(R, vR, vT)) < 10.0**-6.0
    ), "shudf's dlnfdvT does not work"
    return None


def test_estimatemeanvR():
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    vrp8 = dfc.meanvR(0.8)
    assert (
        numpy.fabs(dfc._estimatemeanvR(0.8) - vrp8) < 0.02
    ), "_estimatemeanvR does not agree with meanvR to the expected level"
    return None


def test_asymmetricdrift_flat():
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    vtp8 = dfc.meanvT(0.8)
    assert (
        numpy.fabs(dfc.asymmetricdrift(0.8) - 1.0 + vtp8) < 0.02
    ), "asymmetricdrift does not agree with meanvT for flat rotation curve to the expected level"
    assert (
        numpy.fabs(dfc.asymmetricdrift(1.2) - 1.0 + dfc.meanvT(1.2)) < 0.02
    ), "asymmetricdrift does not agree with meanvT for flat rotation curve to the expected level"
    # also test _estimatemeanvT
    assert (
        numpy.fabs(dfc._estimatemeanvT(0.8) - vtp8) < 0.02
    ), "_estimatemeanvT does not agree with meanvT for flat rotation curve to the expected level"
    return None


def test_asymmetricdrift_powerfall():
    beta = -0.2
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    assert (
        numpy.fabs(dfc.asymmetricdrift(0.8) - 0.8**beta + dfc.meanvT(0.8)) < 0.02
    ), "asymmetricdrift does not agree with meanvT for flat rotation curve to the expected level"
    assert (
        numpy.fabs(dfc.asymmetricdrift(1.2) - 1.2**beta + dfc.meanvT(1.2)) < 0.02
    ), "asymmetricdrift does not agree with meanvT for flat rotation curve to the expected level"
    return None


def test_asymmetricdrift_powerrise():
    beta = 0.2
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    assert (
        numpy.fabs(dfc.asymmetricdrift(0.8) - 0.8**beta + dfc.meanvT(0.8)) < 0.02
    ), "asymmetricdrift does not agree with meanvT for flat rotation curve to the expected level"
    assert (
        numpy.fabs(dfc.asymmetricdrift(1.2) - 1.2**beta + dfc.meanvT(1.2)) < 0.02
    ), "asymmetricdrift does not agree with meanvT for flat rotation curve to the expected level"
    return None


def test_estimateSigmaR2():
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    assert (
        numpy.fabs(dfc._estimateSigmaR2(0.8) / dfc.targetSigma2(0.8) - 1.0) < 0.02
    ), "_estimateSigmaR2 does not agree with targetSigma2 to the expected level"
    return None


def test_estimateSigmaT2():
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.02))
    assert (
        numpy.fabs(dfc._estimateSigmaT2(0.8) / dfc.targetSigma2(0.8) * 2.0 - 1.0) < 0.02
    ), "_estimateSigmaT2 does not agree with targetSigma2 to the expected level"
    assert (
        numpy.fabs(
            dfc._estimateSigmaT2(0.8, log=True)
            - numpy.log(dfc.targetSigma2(0.8))
            + numpy.log(2.0)
        )
        < 0.02
    ), "_estimateSigmaT2 does not agree with targetSigma2 to the expected level"
    return None


def test_vmomentsurfacedensity_deriv():
    # Quick test that the phi derivative is zero
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.02))
    assert (
        numpy.fabs(dfc.vmomentsurfacemass(0.9, 0, 0, deriv="phi")) < 10.0**-6.0
    ), "surfacemass phi derivative is not zero"
    return None


def test_ELtowRRapRperi_flat():
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    Rc = 0.8
    Lc = Rc
    Ec = numpy.log(Rc) + Lc**2.0 / 2.0 / Rc**2.0 + 0.01**2.0 / 2.0
    wr, rap, rperi = dfc._ELtowRRapRperi(Ec, Lc)
    assert (
        numpy.fabs(wr - numpy.sqrt(2.0) / Rc) < 10.0**-3.0
    ), "diskdf's _ELtowRRapRperi's radial frequency for close to circular orbit is wrong"
    return None


def test_ELtowRRapRperi_powerfall():
    beta = -0.2
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    Rc = 0.8
    Lc = Rc * Rc**beta
    Ec = (
        1.0 / 2.0 / beta * Rc ** (2.0 * beta)
        + Lc**2.0 / 2.0 / Rc**2.0
        + 0.01**2.0 / 2.0
    )
    gamma = numpy.sqrt(2.0 / (1.0 + beta))
    wr, rap, rperi = dfc._ELtowRRapRperi(Ec, Lc)
    assert (
        numpy.fabs(wr - 2.0 * Rc ** (beta - 1.0) / gamma) < 10.0**-3.0
    ), "diskdf's _ELtowRRapRperi's radial frequency for close to circular orbit is wrong"
    return None


def test_ELtowRRapRperi_powerrise():
    beta = 0.2
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    Rc = 0.8
    Lc = Rc * Rc**beta
    Ec = (
        1.0 / 2.0 / beta * Rc ** (2.0 * beta)
        + Lc**2.0 / 2.0 / Rc**2.0
        + 0.01**2.0 / 2.0
    )
    gamma = numpy.sqrt(2.0 / (1.0 + beta))
    wr, rap, rperi = dfc._ELtowRRapRperi(Ec, Lc)
    assert (
        numpy.fabs(wr - 2.0 * Rc ** (beta - 1.0) / gamma) < 10.0**-3.0
    ), "diskdf's _ELtowRRapRperi's radial frequency for close to circular orbit is wrong"
    return None


def test_sampledSurfacemassLOS_target():
    numpy.random.seed(1)
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Sample a large number of points, then check some moments against the analytic distribution
    ds = dfc.sampledSurfacemassLOS(numpy.pi / 4.0, n=10000, target=True)
    xds = numpy.linspace(0.001, 4.0, 201)
    pds = numpy.array([dfc.surfacemassLOS(d, 45.0, deg=True, target=True) for d in xds])
    md = numpy.sum(xds * pds) / numpy.sum(pds)
    sd = numpy.sqrt(numpy.sum(xds**2.0 * pds) / numpy.sum(pds) - md**2.0)
    assert (
        numpy.fabs(numpy.mean(ds) - md) < 10.0**-2.0
    ), "mean of surfacemassLOS for target surfacemass is not equal to the mean of the samples"
    assert (
        numpy.fabs(numpy.std(ds) - sd) < 10.0**-2.0
    ), "stddev of surfacemassLOS for target surfacemass is not equal to the mean of the samples"
    assert (
        numpy.fabs(skew_samples(ds) - skew_pdist(xds, pds)) < 10.0**-1
    ), "skew of surfacemassLOS for target surfacemass is not equal to the mean of the samples"
    return None


def test_sampledSurfacemassLOS():
    numpy.random.seed(1)
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Sample a large number of points, then check some moments against the analytic distribution
    ds = dfc.sampledSurfacemassLOS(numpy.pi / 4.0, n=10000, target=False)
    xds = numpy.linspace(0.001, 4.0, 101)
    # check against target, bc that's easy to calculate
    pds = numpy.array([dfc.surfacemassLOS(d, 45.0, deg=True, target=True) for d in xds])
    md = numpy.sum(xds * pds) / numpy.sum(pds)
    sd = numpy.sqrt(numpy.sum(xds**2.0 * pds) / numpy.sum(pds) - md**2.0)
    assert (
        numpy.fabs(numpy.mean(ds) - md) < 10.0**-2.0
    ), "mean of surfacemassLOS surfacemass is not equal to the mean of the samples"
    assert (
        numpy.fabs(numpy.std(ds) - sd) < 10.0**-2.0
    ), "stddev of surfacemassLOS surfacemass is not equal to the mean of the samples"
    assert (
        numpy.fabs(skew_samples(ds) - skew_pdist(xds, pds)) < 10.0**-1
    ), "skew of surfacemassLOS surfacemass is not equal to the mean of the samples"
    return None


def test_sampleVRVT_target_flat():
    numpy.random.seed(1)
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Sample a large number of points, then check some moments against the analytic distribution
    vrvt = dfc.sampleVRVT(0.7, n=500, target=True)
    assert (
        numpy.fabs(numpy.mean(vrvt[:, 0])) < 0.05
    ), "mean vr of vrvt samples is not zero"
    assert (
        numpy.fabs(numpy.mean(vrvt[:, 1]) - dfc.meanvT(0.7)) < 10.0**-2.0
    ), "mean vt of vrvt samples is not equal to numerical calculation"
    assert (
        numpy.fabs(numpy.std(vrvt[:, 0]) - numpy.sqrt(dfc.sigmaR2(0.7))) < 10.0**-1.5
    ), "std dev vr of vrvt samples is not equal to the expected valueo"
    assert (
        numpy.fabs(numpy.std(vrvt[:, 1]) - numpy.sqrt(dfc.sigmaT2(0.7))) < 10.0**-1.5
    ), "std dev vr of vrvt samples is not equal to the expected valueo"
    return None


def test_sampleVRVT_flat():
    numpy.random.seed(1)
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Sample a large number of points, then check some moments against the analytic distribution
    vrvt = dfc.sampleVRVT(0.7, n=500, target=False)
    assert (
        numpy.fabs(numpy.mean(vrvt[:, 0])) < 0.05
    ), "mean vr of vrvt samples is not zero"
    assert (
        numpy.fabs(numpy.mean(vrvt[:, 1]) - dfc.meanvT(0.7)) < 10.0**-2.0
    ), "mean vt of vrvt samples is not equal to numerical calculation"
    assert (
        numpy.fabs(numpy.std(vrvt[:, 0]) - numpy.sqrt(dfc.sigmaR2(0.7))) < 10.0**-1.5
    ), "std dev vr of vrvt samples is not equal to the expected valueo"
    assert (
        numpy.fabs(numpy.std(vrvt[:, 1]) - numpy.sqrt(dfc.sigmaT2(0.7))) < 10.0**-1.5
    ), "std dev vr of vrvt samples is not equal to the expected valueo"
    return None


def test_sampleVRVT_target_powerfall():
    numpy.random.seed(1)
    beta = -0.2
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Sample a large number of points, then check some moments against the analytic distribution
    vrvt = dfc.sampleVRVT(0.7, n=500, target=True)
    assert (
        numpy.fabs(numpy.mean(vrvt[:, 0])) < 0.05
    ), "mean vr of vrvt samples is not zero"
    assert (
        numpy.fabs(numpy.mean(vrvt[:, 1]) - dfc.meanvT(0.7)) < 10.0**-2.0
    ), "mean vt of vrvt samples is not equal to numerical calculation"
    assert (
        numpy.fabs(numpy.std(vrvt[:, 0]) - numpy.sqrt(dfc.sigmaR2(0.7))) < 10.0**-1.5
    ), "std dev vr of vrvt samples is not equal to the expected valueo"
    assert (
        numpy.fabs(numpy.std(vrvt[:, 1]) - numpy.sqrt(dfc.sigmaT2(0.7))) < 10.0**-1.5
    ), "std dev vr of vrvt samples is not equal to the expected valueo"
    return None


def test_sampleLOS_target():
    numpy.random.seed(1)
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Sample a large number of points, then check some moments against the analytic distribution
    os = dfc.sampleLOS(
        numpy.pi / 4.0, n=1000, targetSurfmass=True, targetSigma2=True, deg=False
    )
    ds = numpy.array([o.dist(ro=1.0, obs=[1.0, 0.0, 0.0]) for o in os])
    xds = numpy.linspace(0.001, 4.0, 201)
    pds = numpy.array([dfc.surfacemassLOS(d, 45.0, deg=True, target=True) for d in xds])
    md = numpy.sum(xds * pds) / numpy.sum(pds)
    sd = numpy.sqrt(numpy.sum(xds**2.0 * pds) / numpy.sum(pds) - md**2.0)
    assert (
        numpy.fabs(numpy.mean(ds) - md) < 10.0**-2.0
    ), "mean of distance in sampleLOS for target surfacemass is not equal to the mean of the distribution"
    assert (
        numpy.fabs(numpy.std(ds) - sd) < 10.0**-1.0
    ), "stddev of distance in sampleLOS for target surfacemass is not equal to the mean of the distribution"
    assert (
        numpy.fabs(skew_samples(ds) - skew_pdist(xds, pds)) < 0.3
    ), "skew of distance in sampleLOS for target surfacemass is not equal to the mean of the distribution"
    return None


def test_sampleLOS():
    numpy.random.seed(1)
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Sample a large number of points, then check some moments against the analytic distribution
    os = dfc.sampleLOS(45.0, n=1000, targetSurfmass=False, deg=True)
    ds = numpy.array([o.dist(ro=1.0, obs=[1.0, 0.0, 0.0]) for o in os])
    xds = numpy.linspace(0.001, 4.0, 101)
    # check against target, bc that's easy to calculate
    pds = numpy.array([dfc.surfacemassLOS(d, 45.0, deg=True, target=True) for d in xds])
    md = numpy.sum(xds * pds) / numpy.sum(pds)
    sd = numpy.sqrt(numpy.sum(xds**2.0 * pds) / numpy.sum(pds) - md**2.0)
    assert (
        numpy.fabs(numpy.mean(ds) - md) < 10.0**-2.0
    ), "mean of ds of sampleLOS is not equal to the mean of the distribution"
    assert (
        numpy.fabs(numpy.std(ds) - sd) < 0.05
    ), "stddev of ds of sampleLOS is not equal to the mean of the distribution"
    assert (
        numpy.fabs(skew_samples(ds) - skew_pdist(xds, pds)) < 0.3
    ), "skew of ds of sampleLOS is not equal to the mean of the distribution"
    return None


def test_dehnendf_sample_sampleLOS():
    # Test that the samples returned through sample with los are the same as those returned with sampleLOS
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Sample a large number of points, then check some moments against the analytic distribution
    numpy.random.seed(1)
    os = dfc.sampleLOS(45.0, n=2, targetSurfmass=False, deg=True)
    rs = numpy.array([o.R() for o in os])
    vrs = numpy.array([o.vR() for o in os])
    vts = numpy.array([o.vT() for o in os])
    numpy.random.seed(1)
    os2 = dfc.sample(los=45.0, n=2, targetSurfmass=False, losdeg=True)
    rs2 = numpy.array([o.R() for o in os2])
    vrs2 = numpy.array([o.vR() for o in os2])
    vts2 = numpy.array([o.vT() for o in os2])
    assert numpy.all(
        numpy.fabs(rs - rs2) < 10.0**-10.0
    ), "Samples returned from dehnendf.sample with los set are not the same as those returned with sampleLOS"
    assert numpy.all(
        numpy.fabs(vrs - vrs2) < 10.0**-10.0
    ), "Samples returned from dehnendf.sample with los set are not the same as those returned with sampleLOS"
    assert numpy.all(
        numpy.fabs(vts - vts2) < 10.0**-10.0
    ), "Samples returned from dehnendf.sample with los set are not the same as those returned with sampleLOS"
    return None


def test_shudf_sample_sampleLOS():
    # Test that the samples returned through sample with los are the same as those returned with sampleLOS
    beta = 0.0
    dfc = shudf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    # Sample a large number of points, then check some moments against the analytic distribution
    numpy.random.seed(1)
    os = dfc.sampleLOS(45.0, n=2, targetSurfmass=False, deg=True)
    rs = numpy.array([o.R() for o in os])
    vrs = numpy.array([o.vR() for o in os])
    vts = numpy.array([o.vT() for o in os])
    numpy.random.seed(1)
    os2 = dfc.sample(los=45.0, n=2, targetSurfmass=False, losdeg=True)
    rs2 = numpy.array([o.R() for o in os2])
    vrs2 = numpy.array([o.vR() for o in os2])
    vts2 = numpy.array([o.vT() for o in os2])
    assert numpy.all(
        numpy.fabs(rs - rs2) < 10.0**-10.0
    ), "Samples returned from dehnendf.sample with los set are not the same as those returned with sampleLOS"
    assert numpy.all(
        numpy.fabs(vrs - vrs2) < 10.0**-10.0
    ), "Samples returned from dehnendf.sample with los set are not the same as those returned with sampleLOS"
    assert numpy.all(
        numpy.fabs(vts - vts2) < 10.0**-10.0
    ), "Samples returned from dehnendf.sample with los set are not the same as those returned with sampleLOS"
    return None


def test_dehnendf_sample_flat_returnROrbit():
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    numpy.random.seed(1)
    os = dfc.sample(n=300, returnROrbit=True)
    # Test the spatial distribution
    rs = numpy.array([o.R() for o in os])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.5) < 0.05
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - numpy.sqrt(2.0) / 4.0) < 0.03
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    # Test the velocity distribution
    vrs = numpy.array([o.vR() for o in os])
    assert (
        numpy.fabs(numpy.mean(vrs)) < 0.05
    ), "mean vR of sampled points does not agree with that of the input surface profile (i.e., it is not zero)"
    vts = numpy.array([o.vT() for o in os])
    dvts = numpy.array(
        [vt - r**beta + dfc.asymmetricdrift(r) for (r, vt) in zip(rs, vts)]
    )
    assert (
        numpy.fabs(numpy.mean(dvts)) < 0.1
    ), "mean vT of sampled points does not agree with an estimate based on asymmetric drift"
    return None


def test_dehnendf_sample_flat_returnROrbit_rrange():
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    numpy.random.seed(1)
    os = dfc.sample(n=100, returnROrbit=True, rrange=[0.0, 1.0])
    # Test the spatial distribution
    rs = numpy.array([o.R() for o in os])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.419352) < 0.05
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - 0.240852) < 0.05
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    # Test the velocity distribution
    vrs = numpy.array([o.vR() for o in os])
    assert (
        numpy.fabs(numpy.mean(vrs)) < 0.075
    ), "mean vR of sampled points does not agree with that of the input surface profile (i.e., it is not zero)"
    vts = numpy.array([o.vT() for o in os])
    dvts = numpy.array(
        [vt - r**beta + dfc.asymmetricdrift(r) for (r, vt) in zip(rs, vts)]
    )
    assert (
        numpy.fabs(numpy.mean(dvts)) < 0.1
    ), "mean vT of sampled points does not agree with an estimate based on asymmetric drift"
    return None


def test_dehnendf_sample_powerrise_returnROrbit():
    beta = 0.2
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    numpy.random.seed(1)
    os = dfc.sample(n=300, returnROrbit=True)
    # Test the spatial distribution
    rs = numpy.array([o.R() for o in os])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.5) < 0.1
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - numpy.sqrt(2.0) / 4.0) < 0.06
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    # Test the velocity distribution
    vrs = numpy.array([o.vR() for o in os])
    assert (
        numpy.fabs(numpy.mean(vrs)) < 0.05
    ), "mean vR of sampled points does not agree with that of the input surface profile (i.e., it is not zero)"
    vts = numpy.array([o.vT() for o in os])
    dvts = numpy.array(
        [vt - r**beta + dfc.asymmetricdrift(r) for (r, vt) in zip(rs, vts)]
    )
    assert (
        numpy.fabs(numpy.mean(dvts)) < 0.2
    ), "mean vT of sampled points does not agree with an estimate based on asymmetric drift"
    return None


def test_dehnendf_sample_flat_returnOrbit():
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    numpy.random.seed(1)
    os = dfc.sample(n=100, returnOrbit=True)
    # Test the spatial distribution
    rs = numpy.array([o.R() for o in os])
    phis = numpy.array([o.phi() for o in os])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.5) < 0.05
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.mean(phis) - numpy.pi) < 0.2
    ), "mean phi of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - numpy.sqrt(2.0) / 4.0) < 0.03
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(phis) - numpy.pi / numpy.sqrt(3.0)) < 0.1
    ), "stddev phi of sampled points does not agree with that of the input surface profile"
    # Test the velocity distribution
    vrs = numpy.array([o.vR() for o in os])
    assert (
        numpy.fabs(numpy.mean(vrs)) < 0.05
    ), "mean vR of sampled points does not agree with that of the input surface profile (i.e., it is not zero)"
    vts = numpy.array([o.vT() for o in os])
    dvts = numpy.array(
        [vt - r**beta + dfc.asymmetricdrift(r) for (r, vt) in zip(rs, vts)]
    )
    assert (
        numpy.fabs(numpy.mean(dvts)) < 0.1
    ), "mean vT of sampled points does not agree with an estimate based on asymmetric drift"
    return None


def test_dehnendf_sample_flat_EL():
    beta = 0.0
    dfc = dehnendf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    numpy.random.seed(1)
    EL = dfc.sample(n=50, returnROrbit=False, returnOrbit=False)
    E = [el[0] for el in EL]
    L = [el[1] for el in EL]
    # radii of circular orbits with this energy, these should follow an exponential
    rs = numpy.array([numpy.exp(e - 0.5) for e in E])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.5) < 0.05
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - numpy.sqrt(2.0) / 4.0) < 0.03
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    # BOVY: Could use another test
    return None


def test_shudf_sample_flat_returnROrbit():
    beta = 0.0
    dfc = shudf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    numpy.random.seed(1)
    os = dfc.sample(n=50, returnROrbit=True)
    # Test the spatial distribution
    rs = numpy.array([o.R() for o in os])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.5) < 0.1
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - numpy.sqrt(2.0) / 4.0) < 0.1
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    # Test the velocity distribution
    vrs = numpy.array([o.vR() for o in os])
    assert (
        numpy.fabs(numpy.mean(vrs)) < 0.05
    ), "mean vR of sampled points does not agree with that of the input surface profile (i.e., it is not zero)"
    vts = numpy.array([o.vT() for o in os])
    dvts = numpy.array(
        [vt - r**beta + dfc.asymmetricdrift(r) for (r, vt) in zip(rs, vts)]
    )
    assert (
        numpy.fabs(numpy.mean(dvts)) < 0.1
    ), "mean vT of sampled points does not agree with an estimate based on asymmetric drift"
    return None


def test_shudf_sample_flat_returnROrbit_rrange():
    beta = 0.0
    dfc = shudf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    numpy.random.seed(1)
    os = dfc.sample(n=100, returnROrbit=True, rrange=[0.0, 1.0])
    # Test the spatial distribution
    rs = numpy.array([o.R() for o in os])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.419352) < 0.05
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - 0.240852) < 0.05
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    # Test the velocity distribution
    vrs = numpy.array([o.vR() for o in os])
    assert (
        numpy.fabs(numpy.mean(vrs)) < 0.075
    ), "mean vR of sampled points does not agree with that of the input surface profile (i.e., it is not zero)"
    vts = numpy.array([o.vT() for o in os])
    dvts = numpy.array(
        [vt - r**beta + dfc.asymmetricdrift(r) for (r, vt) in zip(rs, vts)]
    )
    assert (
        numpy.fabs(numpy.mean(dvts)) < 0.13
    ), "mean vT of sampled points does not agree with an estimate based on asymmetric drift"
    return None


def test_shudf_sample_powerrise_returnROrbit():
    beta = 0.2
    dfc = shudf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    numpy.random.seed(1)
    os = dfc.sample(n=100, returnROrbit=True)
    # Test the spatial distribution
    rs = numpy.array([o.R() for o in os])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.5) < 0.1
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - numpy.sqrt(2.0) / 4.0) < 0.06
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    # Test the velocity distribution
    vrs = numpy.array([o.vR() for o in os])
    assert (
        numpy.fabs(numpy.mean(vrs)) < 0.05
    ), "mean vR of sampled points does not agree with that of the input surface profile (i.e., it is not zero)"
    vts = numpy.array([o.vT() for o in os])
    dvts = numpy.array(
        [vt - r**beta + dfc.asymmetricdrift(r) for (r, vt) in zip(rs, vts)]
    )
    assert (
        numpy.fabs(numpy.mean(dvts)) < 0.2
    ), "mean vT of sampled points does not agree with an estimate based on asymmetric drift"
    return None


def test_shudf_sample_flat_returnOrbit():
    beta = 0.0
    dfc = shudf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    numpy.random.seed(1)
    os = dfc.sample(n=100, returnOrbit=True)
    # Test the spatial distribution
    rs = numpy.array([o.R() for o in os])
    phis = numpy.array([o.phi() for o in os])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.5) < 0.05
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.mean(phis) - numpy.pi) < 0.2
    ), "mean phi of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - numpy.sqrt(2.0) / 4.0) < 0.03
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(phis) - numpy.pi / numpy.sqrt(3.0)) < 0.2
    ), "stddev phi of sampled points does not agree with that of the input surface profile"
    # Test the velocity distribution
    vrs = numpy.array([o.vR() for o in os])
    assert (
        numpy.fabs(numpy.mean(vrs)) < 0.05
    ), "mean vR of sampled points does not agree with that of the input surface profile (i.e., it is not zero)"
    vts = numpy.array([o.vT() for o in os])
    dvts = numpy.array(
        [vt - r**beta + dfc.asymmetricdrift(r) for (r, vt) in zip(rs, vts)]
    )
    assert (
        numpy.fabs(numpy.mean(dvts)) < 0.1
    ), "mean vT of sampled points does not agree with an estimate based on asymmetric drift"
    return None


def test_shudf_sample_flat_EL():
    beta = 0.0
    dfc = shudf(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    numpy.random.seed(1)
    EL = dfc.sample(n=50, returnROrbit=False, returnOrbit=False)
    E = [el[0] for el in EL]
    L = [el[1] for el in EL]
    # radii of circular orbits with this angular momentum, these should follow an exponential
    rs = numpy.array(L)
    assert (
        numpy.fabs(numpy.mean(rs) - 0.5) < 0.05
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - numpy.sqrt(2.0) / 4.0) < 0.03
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    # BOVY: Could use another test
    return None


def test_schwarzschild_vs_shu_flat():
    # Schwarzschild DF should be ~~ Shu for small sigma, test w/ flat rotcurve
    dfs = shudf(profileParams=(0.3333333333333333, 1.0, 0.05), beta=0.0, correct=False)
    dfw = schwarzschilddf(
        profileParams=(0.3333333333333333, 1.0, 0.05), beta=0.0, correct=False
    )
    assert (
        numpy.fabs(dfs.meanvT(0.97) - dfw.meanvT(0.97)) < 10.0**-3.0
    ), "Shu and Schwarschild DF differ more than expected for small sigma"
    assert (
        numpy.fabs(dfs.oortA(0.97) - dfw.oortA(0.97)) < 10.0**-2.9
    ), "Shu and Schwarschild DF differ more than expected for small sigma"
    return None


def test_schwarzschild_vs_shu_powerfall():
    # Schwarzschild DF should be ~~ Shu for small sigma, test w/ flat rotcurve
    beta = -0.2
    dfs = shudf(profileParams=(0.3333333333333333, 1.0, 0.05), beta=beta, correct=False)
    dfw = schwarzschilddf(
        profileParams=(0.3333333333333333, 1.0, 0.05), beta=beta, correct=False
    )
    assert (
        numpy.fabs(dfs.meanvT(0.97) - dfw.meanvT(0.97)) < 10.0**-3.0
    ), "Shu and Schwarschild DF differ more than expected for small sigma"
    assert (
        numpy.fabs(dfs.oortA(0.97) - dfw.oortA(0.97)) < 10.0**-3.0
    ), "Shu and Schwarschild DF differ more than expected for small sigma"
    return None


def test_schwarzschild_vs_shu_powerrise():
    # Schwarzschild DF should be ~~ Shu for small sigma, test w/ flat rotcurve
    beta = 0.2
    dfs = shudf(profileParams=(0.3333333333333333, 1.0, 0.05), beta=beta, correct=False)
    dfw = schwarzschilddf(
        profileParams=(0.3333333333333333, 1.0, 0.05), beta=beta, correct=False
    )
    assert (
        numpy.fabs(dfs.meanvT(0.97) - dfw.meanvT(0.97)) < 10.0**-3.0
    ), "Shu and Schwarschild DF differ more than expected for small sigma"
    assert (
        numpy.fabs(dfs.oortA(0.97) - dfw.oortA(0.97)) < 10.0**-2.8
    ), "Shu and Schwarschild DF differ more than expected for small sigma"
    return None


###############################################################################
# Tests of DFcorrection
###############################################################################
def test_dehnendf_flat_DFcorrection_setup():
    global ddf_correct_flat
    global ddf_correct2_flat
    ddf_correct_flat = dehnendf(
        beta=0.0,
        profileParams=(1.0 / 4.0, 1.0, 0.2),
        correct=True,
        niter=1,
        npoints=21,
        savedir=".",
    )
    ddf_correct2_flat = dehnendf(
        beta=0.0,
        profileParams=(1.0 / 4.0, 1.0, 0.2),
        correct=True,
        niter=2,
        npoints=21,
        savedir=".",
    )
    return None


def test_dehnendf_flat_DFcorrection_mag():
    # Test that the call is not too different from before
    tcorr = ddf_correct2_flat._corr.correct(1.1, log=True)
    assert numpy.fabs(tcorr[0]) < 0.15, "dehnendf correction is larger than expected"
    assert numpy.fabs(tcorr[1]) < 0.1, "dehnendf correction is larger than expected"
    # small R
    tcorr = numpy.log(ddf_correct2_flat._corr.correct(10.0**-12.0))
    assert numpy.fabs(tcorr[0]) < 0.4, "dehnendf correction is larger than expected"
    assert numpy.fabs(tcorr[1]) < 1.0, "dehnendf correction is larger than expected"
    # large R
    tcorr = numpy.log(ddf_correct2_flat._corr.correct(12.0))
    assert numpy.fabs(tcorr[0]) < 0.01, "dehnendf correction is larger than expected"
    assert numpy.fabs(tcorr[1]) < 0.01, "dehnendf correction is larger than expected"
    # small R, array
    tcorr = numpy.log(ddf_correct2_flat._corr.correct(10.0**-12.0 * numpy.ones(2)))
    assert numpy.all(
        numpy.fabs(tcorr[0]) < 0.4
    ), "dehnendf correction is larger than expected"
    assert numpy.all(
        numpy.fabs(tcorr[1]) < 1.0
    ), "dehnendf correction is larger than expected"
    # large R
    tcorr = numpy.log(ddf_correct2_flat._corr.correct(12.0 * numpy.ones(2)))
    assert numpy.all(
        numpy.fabs(tcorr[0]) < 0.01
    ), "dehnendf correction is larger than expected"
    assert numpy.all(
        numpy.fabs(tcorr[1]) < 0.01
    ), "dehnendf correction is larger than expected"
    # large R, log
    tcorr = ddf_correct2_flat._corr.correct(12.0 * numpy.ones(2), log=True)
    assert numpy.all(
        numpy.fabs(tcorr[0]) < 0.01
    ), "dehnendf correction is larger than expected"
    assert numpy.all(
        numpy.fabs(tcorr[1]) < 0.01
    ), "dehnendf correction is larger than expected"
    return None


def test_dehnendf_flat_DFcorrection_deriv_mag():
    # Test that the derivative behaves as expected
    tcorr = ddf_correct2_flat._corr.derivLogcorrect(2.0)
    assert (
        numpy.fabs(tcorr[0]) < 0.1
    ), "dehnendf correction derivative is larger than expected"
    assert (
        numpy.fabs(tcorr[1]) < 0.1
    ), "dehnendf correction derivative is larger than expected"
    # small R, derivative should be very large
    tcorr = ddf_correct2_flat._corr.derivLogcorrect(10.0**-12.0)
    assert (
        numpy.fabs(tcorr[0]) > 1.0
    ), "dehnendf correction derivative is smaller than expected"
    assert (
        numpy.fabs(tcorr[1]) > 1.0
    ), "dehnendf correction derivative is larger than expected"
    # large R
    tcorr = ddf_correct2_flat._corr.derivLogcorrect(12.0)
    assert (
        numpy.fabs(tcorr[0]) < 0.01
    ), "dehnendf correction derivative is larger than expected"
    assert (
        numpy.fabs(tcorr[1]) < 0.01
    ), "dehnendf correction derivative is larger than expected"
    return None


def test_dehnendf_flat_DFcorrection_surfacemass():
    # Test that the surfacemass is better than before
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2), correct=False)
    diff_uncorr = numpy.fabs(
        numpy.log(dfc.surfacemass(0.8)) - numpy.log(dfc.targetSurfacemass(0.8))
    )
    diff_corr = numpy.fabs(
        numpy.log(ddf_correct_flat.surfacemass(0.8))
        - numpy.log(dfc.targetSurfacemass(0.8))
    )
    diff_corr2 = numpy.fabs(
        numpy.log(ddf_correct2_flat.surfacemass(0.8))
        - numpy.log(dfc.targetSurfacemass(0.8))
    )
    assert (
        diff_corr < diff_uncorr
    ), "surfacemass w/ corrected dehnenDF is does not agree better with target than with uncorrected dehnenDF"
    assert (
        diff_corr2 < diff_corr
    ), "surfacemass w/ corrected dehnenDF w/ 2 iterations is does not agree better with target than with 1 iteration"
    return None


def test_dehnendf_flat_DFcorrection_sigmaR2():
    # Test that the sigmaR2 is better than before
    dfc = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2), correct=False)
    diff_uncorr = numpy.fabs(
        numpy.log(dfc.sigmaR2(0.8)) - numpy.log(dfc.targetSigma2(0.8))
    )
    diff_corr = numpy.fabs(
        numpy.log(ddf_correct_flat.sigmaR2(0.8)) - numpy.log(dfc.targetSigma2(0.8))
    )
    diff_corr2 = numpy.fabs(
        numpy.log(ddf_correct2_flat.sigmaR2(0.8)) - numpy.log(dfc.targetSigma2(0.8))
    )
    assert (
        diff_corr < diff_uncorr
    ), "sigmaR2 w/ corrected dehnenDF is does not agree better with target than with uncorrected dehnenDF"
    assert (
        diff_corr2 < diff_corr
    ), "sigmaR2 w/ corrected dehnenDF w/ 2 iterations is does not agree better with target than with 1 iteration"
    return None


def test_dehnendf_flat_DFcorrection_reload():
    # Test that the corrections aren't re-calculated if they were saved
    import time

    start = time.time()
    reddf = dehnendf(
        beta=0.0,
        profileParams=(1.0 / 4.0, 1.0, 0.2),
        correct=True,
        niter=1,
        npoints=21,
        savedir=".",
    )
    assert (
        time.time() - start < 1.0
    ), "Setup w/ correct=True, but already computed corrections takes too long"
    return None


def test_dehnendf_flat_DFcorrection_cleanup():
    # This should run quickly
    dfc = dehnendf(
        beta=0.0,
        profileParams=(1.0 / 4.0, 1.0, 0.2),
        correct=True,
        niter=1,
        npoints=21,
        savedir=".",
    )
    try:
        os.remove(dfc._corr._createSavefilename(1))
    except:
        raise AssertionError("removing DFcorrection's savefile did not work")
    try:
        os.remove(dfc._corr._createSavefilename(2))
    except:
        raise AssertionError("removing DFcorrection's savefile did not work")
    return None


def test_DFcorrection_setup():
    # Test that the keywords are setup correctly and that exceptions are raised
    dfc = dehnendf(
        beta=0.1,
        profileParams=(1.0 / 3.0, 1.0, 0.2),
        correct=True,
        rmax=4.0,
        niter=2,
        npoints=5,
        interp_k=3,
        savedir=".",
    )
    assert (
        numpy.fabs(dfc._corr._rmax - 4.0) < 10.0**-10.0
    ), "rmax not set up correctly in DFcorrection"
    assert (
        numpy.fabs(dfc._corr._npoints - 5) < 10.0**-10.0
    ), "npoints not set up correctly in DFcorrection"
    assert (
        numpy.fabs(dfc._corr._interp_k - 3) < 10.0**-10.0
    ), "interp_k not set up correctly in DFcorrection"
    assert (
        numpy.fabs(dfc._corr._beta - 0.1) < 10.0**-10.0
    ), "beta not set up correctly in DFcorrection"
    # setup w/ corrections
    corrs = dfc._corr._corrections
    dfc = dehnendf(
        beta=0.1,
        profileParams=(1.0 / 3.0, 1.0, 0.2),
        correct=True,
        rmax=4.0,
        niter=2,
        interp_k=3,
        savedir=".",
        corrections=corrs,
    )
    assert numpy.all(
        numpy.fabs(corrs - dfc._corr._corrections) < 10.0**-10.0
    ), "DFcorrection initialized w/ corrections does not work properly"
    # If corrections.shape[0] neq npoints, should raise error
    from galpy.df.diskdf import DFcorrectionError

    try:
        dfc = dehnendf(
            beta=0.1,
            profileParams=(1.0 / 3.0, 1.0, 0.2),
            correct=True,
            rmax=4.0,
            niter=2,
            npoints=6,
            interp_k=3,
            savedir=".",
            corrections=corrs,
        )
    except DFcorrectionError:
        pass
    else:
        raise AssertionError(
            "DFcorrection setup with corrections.shape[0] neq npoints did not raise DFcorrectionError"
        )
    # rm savefile
    dfc = dehnendf(
        beta=0.1,
        profileParams=(1.0 / 3.0, 1.0, 0.2),
        correct=True,
        rmax=4.0,
        niter=2,
        npoints=5,
        interp_k=3,
        savedir=".",
    )
    try:
        os.remove(dfc._corr._createSavefilename(2))
    except:
        raise AssertionError("removing DFcorrection's savefile did not work")
    # Also explicily setup a DFcorrection, to test for other stuff
    from galpy.df import DFcorrection
    from galpy.df.diskdf import DFcorrectionError

    # Should raise DFcorrectionError bc surfaceSigmaProfile is not set
    try:
        dfc = DFcorrection(npoints=2, niter=2, rmax=4.0, beta=-0.1, interp_k=3)
    except DFcorrectionError as e:
        print(e)
    else:
        raise AssertionError(
            "DFcorrection setup with no surfaceSigmaProfile set did not raise DFcorrectionError"
        )
    # Now w/ surfaceSigmaProfile to test default dftype
    from galpy.df import expSurfaceSigmaProfile

    essp = expSurfaceSigmaProfile(params=(0.25, 0.75, 0.1))
    dfc = DFcorrection(
        npoints=5, niter=1, rmax=4.0, surfaceSigmaProfile=essp, interp_k=3
    )
    assert issubclass(
        dfc._dftype, dehnendf
    ), "DFcorrection w/ no dftype set does not default to dehnendf"
    assert (
        numpy.fabs(dfc._beta) < 10.0**-10.0
    ), "DFcorrection w/ no beta does not default to zero"
    try:
        os.remove(dfc._createSavefilename(1))
    except:
        raise AssertionError("removing DFcorrection's savefile did not work")
    return None


def test_dehnendf_sample_flat_returnROrbit_wcorrections():
    beta = 0.0
    dfc = ddf_correct2_flat
    numpy.random.seed(1)
    os = dfc.sample(n=100, returnROrbit=True)
    # Test the spatial distribution
    rs = numpy.array([o.R() for o in os])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.5) < 0.05
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - numpy.sqrt(2.0) / 4.0) < 0.03
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    # Test the velocity distribution
    vrs = numpy.array([o.vR() for o in os])
    assert (
        numpy.fabs(numpy.mean(vrs)) < 0.1
    ), "mean vR of sampled points does not agree with that of the input surface profile (i.e., it is not zero)"
    vts = numpy.array([o.vT() for o in os])
    dvts = numpy.array(
        [vt - r**beta + dfc.asymmetricdrift(r) for (r, vt) in zip(rs, vts)]
    )
    assert (
        numpy.fabs(numpy.mean(dvts)) < 0.1
    ), "mean vT of sampled points does not agree with an estimate based on asymmetric drift"
    return None


def test_shudf_flat_DFcorrection_setup():
    global sdf_correct_flat
    sdf_correct_flat = shudf(
        beta=0.0,
        profileParams=(1.0 / 4.0, 1.0, 0.2),
        correct=True,
        niter=1,
        npoints=21,
        savedir=".",
    )
    return None


def test_shudf_flat_DFcorrection_surfacemass():
    # Test that the surfacemass is better than before
    dfc = shudf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2), correct=False)
    diff_uncorr = numpy.fabs(
        numpy.log(dfc.surfacemass(0.8)) - numpy.log(dfc.targetSurfacemass(0.8))
    )
    diff_corr = numpy.fabs(
        numpy.log(sdf_correct_flat.surfacemass(0.8))
        - numpy.log(dfc.targetSurfacemass(0.8))
    )
    assert (
        diff_corr < diff_uncorr
    ), "surfacemass w/ corrected shuDF is does not agree better with target than with uncorrected shuDF"
    return None


def test_shudf_sample_flat_returnROrbit_wcorrections():
    beta = 0.0
    dfc = sdf_correct_flat
    numpy.random.seed(1)
    os = dfc.sample(n=100, returnROrbit=True)
    # Test the spatial distribution
    rs = numpy.array([o.R() for o in os])
    assert (
        numpy.fabs(numpy.mean(rs) - 0.5) < 0.05
    ), "mean R of sampled points does not agree with that of the input surface profile"
    assert (
        numpy.fabs(numpy.std(rs) - numpy.sqrt(2.0) / 4.0) < 0.035
    ), "stddev R of sampled points does not agree with that of the input surface profile"
    # Test the velocity distribution
    vrs = numpy.array([o.vR() for o in os])
    assert (
        numpy.fabs(numpy.mean(vrs)) < 0.05
    ), "mean vR of sampled points does not agree with that of the input surface profile (i.e., it is not zero)"
    vts = numpy.array([o.vT() for o in os])
    dvts = numpy.array(
        [vt - r**beta + dfc.asymmetricdrift(r) for (r, vt) in zip(rs, vts)]
    )
    assert (
        numpy.fabs(numpy.mean(dvts)) < 0.1
    ), "mean vT of sampled points does not agree with an estimate based on asymmetric drift"
    return None


def test_shudf_flat_DFcorrection_cleanup():
    # This should run quickly
    dfc = shudf(
        beta=0.0,
        profileParams=(1.0 / 4.0, 1.0, 0.2),
        correct=True,
        niter=1,
        npoints=21,
        savedir=".",
    )
    try:
        os.remove(dfc._corr._createSavefilename(1))
    except:
        raise AssertionError("removing DFcorrection's savefile did not work")
    return None


def test_axipotential():
    from galpy.df.diskdf import _RMIN, axipotential

    assert (
        numpy.fabs(axipotential(numpy.array([0.5]), beta=0.0) - numpy.log(0.5))
        < 10.0**-8
    ), "axipotential w/ beta=0.0 does not work as expected"
    assert (
        numpy.fabs(axipotential(numpy.array([0.5]), beta=0.2) - 1.0 / 0.4 * 0.5**0.4)
        < 10.0**-8
    ), "axipotential w/ beta=0.2 does not work as expected"
    assert (
        numpy.fabs(axipotential(numpy.array([0.5]), beta=-0.2) + 1.0 / 0.4 * 0.5**-0.4)
        < 10.0**-8
    ), "axipotential w/ beta=0.2 does not work as expected"
    # special case of R=0 should go to _RMIN
    assert (
        numpy.fabs(axipotential(numpy.array([0.0]), beta=0.0) - numpy.log(_RMIN))
        < 10.0**-8
    ), "axipotential w/ beta=0.0 does not work as expected"
    return None


def test_dlToRphi():
    from galpy.df.diskdf import _dlToRphi

    R, theta = _dlToRphi(1.0, 0.0)
    assert (
        numpy.fabs(R) < 10.0**-3.0
    ), "_dlToRphi close to center does not behave properly"
    assert (
        numpy.fabs(theta % numpy.pi) < 10.0**-3.0
    ), "_dlToRphi close to center does not behave properly"
    return None


def skew_samples(s):
    m1 = numpy.mean(s)
    m2 = numpy.mean((s - m1) ** 2.0)
    m3 = numpy.mean((s - m1) ** 3.0)
    return m3 / m2**1.5


def skew_pdist(s, ps):
    norm = numpy.sum(ps)
    m1 = numpy.sum(s * ps) / norm
    m2 = numpy.sum((s - m1) ** 2.0 * ps) / norm
    m3 = numpy.sum((s - m1) ** 3.0 * ps) / norm
    return m3 / m2**1.5
