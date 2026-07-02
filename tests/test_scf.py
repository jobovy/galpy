############################TESTS ON POTENTIALS################################

import numpy
import pytest

from galpy import df, potential
from galpy.orbit import Orbit
from galpy.potential import SCFPotential
from galpy.util import coords

EPS = 1e-13  ## default epsilon

DEFAULT_R = numpy.array([0.5, 1.0, 2.0])
DEFAULT_Z = numpy.array([0.0, 0.125, -0.125, 0.25, -0.25])
DEFAULT_PHI = numpy.array(
    [0.0, 0.5, -0.5, 1.0, -1.0, numpy.pi, 0.5 + numpy.pi, 1.0 + numpy.pi]
)

##Tests whether invalid coefficients will throw an error at runtime


def test_coeffs_toomanydimensions():
    Acos = numpy.ones((10, 2, 32, 34))
    try:
        SCFPotential(Acos=Acos)
        raise Exception("Expected RuntimeError")
    except RuntimeError:
        pass


def test_coeffs_toolittledimensions():
    Acos = numpy.ones((10, 2))
    try:
        SCFPotential(Acos=Acos)
        raise Exception("Expected RuntimeError")
    except RuntimeError:
        pass


def test_coeffs_AsinNone_LnotequalM():
    Acos = numpy.ones((2, 3, 4))
    try:
        SCFPotential(Acos=Acos)
        raise Exception("Expected RuntimeError")
    except RuntimeError:
        pass


def test_coeffs_AsinNotNone_LnotequalM():
    Acos = numpy.ones((2, 3, 4))
    Asin = numpy.ones((2, 3, 4))

    try:
        SCFPotential(Acos=Acos, Asin=Asin)
        raise Exception("Expected RuntimeError")
    except RuntimeError:
        pass


def test_coeffs_AsinNone_Mequals1():
    Acos = numpy.zeros((2, 3, 1))
    Asin = None

    SCFPotential(Acos=Acos, Asin=Asin)


def test_coeffs_AsinNone_MequalsL():
    Acos = numpy.zeros((2, 3, 3))
    Asin = None

    SCFPotential(Acos=Acos, Asin=Asin)


def test_coeffs_AsinNone_AcosNotaxisym():
    Acos = numpy.ones((2, 3, 3))
    try:
        SCFPotential(Acos=Acos)
        raise Exception("Expected RuntimeError")
    except RuntimeError:
        pass


def test_coeffs_AsinShape_notequal_AcosShape():
    Acos = numpy.ones((2, 3, 3))
    Asin = numpy.ones((2, 2, 2))
    try:
        SCFPotential(Acos=Acos, Asin=Asin)
        raise Exception("Expected RuntimeError")
    except RuntimeError:
        pass


def test_coeffs_Acos_L_M_notLowerTriangular():
    Acos = numpy.ones((2, 3, 3))
    Asin = numpy.zeros((2, 3, 3))
    try:
        SCFPotential(Acos=Acos, Asin=Asin)
        raise Exception("Expected RuntimeWarning")
    except RuntimeWarning:
        pass


def test_coeffs_Asin_L_M_notLowerTriangular():
    Acos = numpy.zeros((2, 3, 3))
    Asin = numpy.ones((2, 3, 3))
    try:
        SCFPotential(Acos=Acos, Asin=Asin)
        raise Exception("Expected RuntimeWarning")
    except RuntimeWarning:
        pass


def testAxi_phiIsNone():
    R = 1
    z = 0
    phi = 1.1
    scf = SCFPotential()
    assert scf(R, z, None) == scf(R, z, phi), (
        "The axisymmetric potential does not work at phi=None"
    )
    assert scf.dens(R, z, None) == scf.dens(R, z, phi), (
        "The axisymmetric density does not work at phi=None"
    )
    assert scf.Rforce(R, z, None) == scf.Rforce(R, z, phi), (
        "The axisymmetric Rforce does not work at phi=None"
    )
    assert scf.zforce(R, z, None) == scf.zforce(R, z, phi), (
        "The axisymmetric zforce does not work at phi=None"
    )
    assert scf.phitorque(R, z, None) == scf.phitorque(R, z, phi), (
        "The axisymmetric phitorque does not work at phi=None"
    )


##Tests user inputs as arrays


def testArray_RArray():
    scf = SCFPotential()
    array = numpy.linspace(0, 3, 100)
    ArrayTest(scf, [array, 1.0, 0])


def testArray_zArray():
    scf = SCFPotential()
    array = numpy.linspace(0, 3, 100)
    ArrayTest(scf, [1.0, array, 0])


def testArray_phiArray():
    scf = SCFPotential()
    array = numpy.linspace(0, 3, 100)
    ArrayTest(scf, [1.0, 1.0, array])


def testArrayBroadcasting():
    scf = SCFPotential()
    R = numpy.ones((10, 20, 2))
    z = numpy.linspace(0, numpy.pi, 10)[:, None, None]
    phi = numpy.zeros((10, 20))[:, :, None]

    ArrayTest(scf, [R, z, phi])


## tests whether scf_compute_spherical computes the correct coefficients for a Hernquist Potential
def test_scf_compute_spherical_hernquist():
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 10)
    spherical_coeffsTest(Acos, Asin)
    assert numpy.fabs(Acos[0, 0, 0] - 1.0) < EPS, (
        f"Acos(n=0,l=0,m=0) = 1 fails. Found to be Acos(n=0,l=0,m=0) = {Acos[0, 0, 0]}"
    )
    assert numpy.all(numpy.fabs(Acos[1:, 0, 0]) < EPS), "Acos(n>0,l=0,m=0) = 0 fails."


## tests whether scf_compute_spherical computes the correct coefficients for Zeeuw's Potential
def test_scf_compute_spherical_zeeuw():
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_Zeeuw, 10)
    spherical_coeffsTest(Acos, Asin)
    assert numpy.fabs(Acos[0, 0, 0] - 2 * 3.0 / 4) < EPS, (
        f"Acos(n=0,l=0,m=0) = 3/2 fails. Found to be Acos(n=0,l=0,m=0) = {Acos[0, 0, 0]}"
    )
    assert numpy.fabs(Acos[1, 0, 0] - 2 * 1.0 / 12) < EPS, (
        f"Acos(n=1,l=0,m=0) = 1/6 fails. Found to be Acos(n=0,l=0,m=0) = {Acos[0, 0, 0]}"
    )
    assert numpy.all(numpy.fabs(Acos[2:, 0, 0]) < EPS), "Acos(n>1,l=0,m=0) = 0 fails."


##Tests that the numerically calculated results from axi_density1 matches with the analytic results
def test_scf_compute_axi_density1():
    A = potential.scf_compute_coeffs_axi(axi_density1, 10, 10)
    axi_coeffsTest(A[0], A[1])
    analytically_calculated = numpy.array(
        [
            [4.0 / 3, 7.0 * 3 ** (-5 / 2.0), 2 * 11 * 5 ** (-5.0 / 2), 0],
            [0, 0, 0, 0],
            [
                0,
                11.0 / (3.0 ** (5.0 / 2) * 5 * 7.0 * 2),
                1.0 / (2 * 3.0 * 5**0.5 * 7.0),
                0,
            ],
        ]
    )
    numerically_calculated = A[0][:3, :4, 0]
    shape = numerically_calculated.shape
    for n in range(shape[0]):
        for l in range(shape[1]):
            assert (
                numpy.fabs(numerically_calculated[n, l] - analytically_calculated[n, l])
                < EPS
            ), (
                f"Acos(n={n},l={l},0) = {numerically_calculated[n, l]}, whereas it was analytically calculated to be {analytically_calculated[n, l]}"
            )
    # Checks that A at l != 0,1,2 are always zero
    assert numpy.all(numpy.fabs(A[0][:, 3:, 0]) < 1e-10), "Acos(n,l>2,m=0) = 0 fails."

    # Checks that A at n odd is always zero
    assert numpy.all(numpy.fabs(A[0][1::2, :, 0]) < 1e-10), (
        "Acos(n odd,l,m=0) = 0 fails."
    )

    # Checks that A = 0 when n != 0 and l = 0
    assert numpy.all(numpy.fabs(A[0][1:, 0, 0]) < 1e-10), (
        "Acos(n > 1,l=0,m=0) = 0 fails."
    )


##Tests that the numerically calculated results from axi_density2 matches with the analytic results
def test_scf_compute_axi_density2():
    A = potential.scf_compute_coeffs_axi(
        axi_density2, 10, 10, radial_order=30, costheta_order=12
    )
    axi_coeffsTest(A[0], A[1])
    analytically_calculated = 2 * numpy.array(
        [
            [1.0, 7.0 * 3 ** (-3 / 2.0) / 4.0, 3 * 11 * 5 ** (-5.0 / 2) / 2.0, 0],
            [0, 0, 0, 0],  ##I never did analytically solve for n=1
            [
                0,
                11.0 / (7 * 5 * 3 ** (3.0 / 2) * 2 ** (3.0)),
                (7 * 5 ** (0.5) * 2**3.0) ** -1.0,
                0,
            ],
        ]
    )
    numerically_calculated = A[0][:3, :4, 0]
    shape = numerically_calculated.shape
    for n in range(shape[0]):
        if n == 1:
            continue
        for l in range(shape[1]):
            assert (
                numpy.fabs(numerically_calculated[n, l] - analytically_calculated[n, l])
                < EPS
            ), (
                f"Acos(n={n},l={l},0) = {numerically_calculated[n, l]}, whereas it was analytically calculated to be {analytically_calculated[n, l]}"
            )

    # Checks that A at l != 0,1,2 are always zero
    assert numpy.all(numpy.fabs(A[0][:, 3:, 0]) < 1e-10), "Acos(n,l>2,m=0) = 0 fails."

    # Checks that A = 0 when n = 2,4,..,2*n and l = 0
    assert numpy.all(numpy.fabs(A[0][2::2, 0, 0]) < 1e-10), (
        "Acos(n > 1,l = 0,m=0) = 0 fails."
    )


## Tests how nbody calculation compares to density calculation for scf_compute_coeff in the spherical case
def test_scf_compute_spherical_nbody_hernquist():
    N = int(1e6)
    Mh = 11.0
    ah = 50.0 / 8.0
    m = Mh / N
    factor = 1.0
    nsamp = 10
    Norder = 10

    hern = potential.HernquistPotential(amp=2 * Mh, a=ah)
    hern.turn_physical_off()
    hdf = df.isotropicHernquistdf(hern)
    numpy.random.seed(1)
    samples = [hdf.sample(n=N) for i in range(nsamp)]

    positions = numpy.array(
        [
            [samples[i].x(), samples[i].y(), samples[i].z() * factor]
            for i in range(nsamp)
        ]
    )

    c = numpy.zeros((nsamp, Norder, 1, 1))
    s = numpy.zeros((nsamp, Norder, 1, 1))
    for i in range(nsamp):
        c[i], s[i] = potential.scf_compute_coeffs_spherical_nbody(
            positions[i], Norder, mass=m * numpy.ones(N), a=ah
        )

    cc, ss = potential.scf_compute_coeffs_spherical(hern.dens, Norder, a=ah)

    # Check that the difference between the coefficients is within the standard deviation
    assert (cc - numpy.mean(c, axis=0) < numpy.std(c, axis=0)).all()

    # Repeat test for single mass
    c = numpy.zeros((nsamp, Norder, 1, 1))
    s = numpy.zeros((nsamp, Norder, 1, 1))
    for i in range(nsamp):
        c[i], s[i] = potential.scf_compute_coeffs_spherical_nbody(
            positions[i], Norder, mass=m, a=ah
        )
    assert (cc - numpy.mean(c, axis=0) < numpy.std(c, axis=0)).all()
    return None


## Tests how nbody calculation compares to density calculation for scf_compute_coeff
def test_scf_compute_axi_nbody_twopowertriaxial():
    N = int(1e5)
    Mh = 11.0
    ah = 50.0 / 8.0
    m = Mh / N
    zfactor = 2.5
    nsamp = 10
    Norder = 10
    Lorder = 10

    hern = potential.HernquistPotential(amp=2 * Mh, a=ah)
    hern.turn_physical_off()
    hdf = df.isotropicHernquistdf(hern)
    numpy.random.seed(1)
    samp = [hdf.sample(n=N) for i in range(nsamp)]

    positions = numpy.array(
        [[samp[i].x(), samp[i].y(), samp[i].z() * zfactor] for i in range(nsamp)]
    )

    # This is an axisymmtric Hernquist profile with the same mass as the above
    tptp = potential.TwoPowerTriaxialPotential(
        amp=2.0 * Mh / zfactor, a=ah, alpha=1.0, beta=4.0, b=1.0, c=zfactor
    )
    tptp.turn_physical_off()

    cc, ss = potential.scf_compute_coeffs_axi(tptp.dens, Norder, Lorder, a=ah)
    c, s = numpy.zeros((2, nsamp, Norder, Lorder, 1))
    for i, p in enumerate(positions):
        c[i], s[i] = potential.scf_compute_coeffs_axi_nbody(
            p, Norder, Lorder, mass=m * numpy.ones(N), a=ah
        )

    # Check that the difference between the coefficients is within two standard deviations
    assert (cc - (numpy.mean(c, axis=0)) <= (2.0 * numpy.std(c, axis=0))).all()

    # Repeat test for single mass
    c, s = numpy.zeros((2, nsamp, Norder, Lorder, 1))
    for i, p in enumerate(positions):
        c[i], s[i] = potential.scf_compute_coeffs_axi_nbody(
            p, Norder, Lorder, mass=m, a=ah
        )
    assert (cc - (numpy.mean(c, axis=0)) <= (2.0 * numpy.std(c, axis=0))).all()
    return None


## Tests how nbody calculation compares to density calculation for scf_compute_coeff
def test_scf_compute_nbody_twopowertriaxial():
    N = int(1e5)
    Mh = 11.0
    ah = 50.0 / 8.0
    m = Mh / N
    yfactor = 1.5
    zfactor = 2.5
    nsamp = 10
    Norder = 10
    Lorder = 10

    hern = potential.HernquistPotential(amp=2 * Mh, a=ah)
    hern.turn_physical_off()
    hdf = df.isotropicHernquistdf(hern)
    numpy.random.seed(2)
    samp = [hdf.sample(n=N) for i in range(nsamp)]

    positions = numpy.array(
        [
            [samp[i].x(), samp[i].y() * yfactor, samp[i].z() * zfactor]
            for i in range(nsamp)
        ]
    )

    # This is an triaxial Hernquist profile with the same mass as the above
    tptp = potential.TwoPowerTriaxialPotential(
        amp=2.0 * Mh / yfactor / zfactor,
        a=ah,
        alpha=1.0,
        beta=4.0,
        b=yfactor,
        c=zfactor,
    )
    tptp.turn_physical_off()

    cc, ss = potential.scf_compute_coeffs(tptp.dens, Norder, Lorder, a=ah)
    c, s = numpy.zeros((2, nsamp, Norder, Lorder, Lorder))
    for i, p in enumerate(positions):
        c[i], s[i] = potential.scf_compute_coeffs_nbody(
            p, Norder, Lorder, mass=m * numpy.ones(N), a=ah
        )

    # Check that the difference between the coefficients is within two standard deviations
    assert (cc - (numpy.mean(c, axis=0)) <= (2.0 * numpy.std(c, axis=0))).all()

    # Repeat test for single mass
    c, s = numpy.zeros((2, nsamp, Norder, Lorder, Lorder))
    for i, p in enumerate(positions):
        c[i], s[i] = potential.scf_compute_coeffs_nbody(p, Norder, Lorder, mass=m, a=ah)
    assert (cc - (numpy.mean(c, axis=0)) <= (2.0 * numpy.std(c, axis=0))).all()
    return None


def test_scf_compute_nfw():
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW, 10)
    spherical_coeffsTest(Acos, Asin)


##Tests radial order from scf_compute_coeffs_spherical
def test_nfw_sphericalOrder():
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW, 10)
    Acos2, Asin2 = potential.scf_compute_coeffs_spherical(rho_NFW, 10, radial_order=50)

    assert numpy.all(numpy.fabs(Acos - Acos2) < EPS), (
        "Increasing the radial order fails for scf_compute_coeffs_spherical"
    )


##Tests radial and costheta order from scf_compute_coeffs_axi
def test_axi_density1_axiOrder():
    Acos, Asin = potential.scf_compute_coeffs_axi(axi_density1, 10, 10)
    Acos2, Asin2 = potential.scf_compute_coeffs_axi(
        axi_density1, 10, 10, radial_order=50, costheta_order=50
    )

    assert numpy.all(numpy.fabs(Acos - Acos2) < 1e-10), (
        "Increasing the radial and costheta order fails for scf_compute_coeffs_axi"
    )


##Tests radial, costheta and phi order from scf_compute_coeffs
def test_density1_Order():
    Acos, Asin = potential.scf_compute_coeffs(density1, 5, 5)
    Acos2, Asin2 = potential.scf_compute_coeffs(
        density1, 5, 5, radial_order=19, costheta_order=19, phi_order=19
    )
    assert numpy.all(numpy.fabs(Acos - Acos2) < 1e-3), (
        "Increasing the radial, costheta, and phi order fails for Acos from scf_compute_coeffs"
    )

    assert numpy.all(numpy.fabs(Asin - Asin) < EPS), (
        "Increasing the radial, costheta, and phi order fails for Asin from scf_compute_coeffs"
    )


## Tests whether scf_compute_axi reduces to scf_compute_spherical for the Hernquist Potential
def test_scf_axiHernquistCoeffs_ReducesToSpherical():
    Aspherical = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 10)
    Aaxi = potential.scf_compute_coeffs_axi(sphericalHernquistDensity, 10, 10)
    axi_reducesto_spherical(Aspherical, Aaxi, "Hernquist Potential")


## Tests whether scf_compute_axi reduces to scf_compute_spherical for Zeeuw's Potential
def test_scf_axiZeeuwCoeffs_ReducesToSpherical():
    Aspherical = potential.scf_compute_coeffs_spherical(rho_Zeeuw, 10)
    Aaxi = potential.scf_compute_coeffs_axi(rho_Zeeuw, 10, 10)
    axi_reducesto_spherical(Aspherical, Aaxi, "Zeeuw Potential")


## Tests whether scf_compute reduces to scf_compute_spherical for Hernquist Potential
def test_scf_HernquistCoeffs_ReducesToSpherical():
    Aspherical = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 5)
    Aaxi = potential.scf_compute_coeffs(sphericalHernquistDensity, 5, 5)
    reducesto_spherical(Aspherical, Aaxi, "Hernquist Potential")


## Tests whether scf_compute reduces to scf_compute_spherical for Zeeuw's Potential
def test_scf_ZeeuwCoeffs_ReducesToSpherical():
    Aspherical = potential.scf_compute_coeffs_spherical(rho_Zeeuw, 5)
    Aaxi = potential.scf_compute_coeffs(
        rho_Zeeuw, 5, 5, radial_order=20, costheta_order=20
    )
    reducesto_spherical(Aspherical, Aaxi, "Zeeuw Potential")


## Tests whether scf density matches with Hernquist density
def test_densMatches_hernquist():
    h = potential.HernquistPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 10)
    scf = SCFPotential()
    assertmsg = "Comparing the density of Hernquist Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(h.dens, scf.dens, assertmsg)


## Tests whether scf density matches with Zeeuw density
def test_densMatches_zeeuw():
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_Zeeuw, 10)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing the density of Zeeuw's perfect ellipsoid with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(rho_Zeeuw, scf.dens, assertmsg)


## Tests whether scf density matches with axi_density1
def test_densMatches_axi_density1():
    Acos, Asin = potential.scf_compute_coeffs_axi(axi_density1, 50, 3)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing axi_density1 with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(axi_density1, scf.dens, assertmsg, eps=1e-3)


## Tests whether scf density matches with axi_density2
def test_densMatches_axi_density2():
    Acos, Asin = potential.scf_compute_coeffs_axi(axi_density2, 50, 3)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing axi_density2 with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(axi_density2, scf.dens, assertmsg, eps=1e-3)


## Tests whether scf density matches with NFW
def test_densMatches_nfw():
    nfw = potential.NFWPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW, 50, a=50)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin, a=50)
    assertmsg = "Comparing nfw with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(nfw.dens, scf.dens, assertmsg, eps=1e-2)


## Tests whether scf potential matches with Hernquist potential
def test_potentialMatches_hernquist():
    h = potential.HernquistPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 10)
    scf = SCFPotential()
    assertmsg = "Comparing the potential of Hernquist Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(h, scf, assertmsg)


## Tests whether scf Potential matches with NFW
def test_potentialMatches_nfw():
    nfw = potential.NFWPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW, 50, a=50)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin, a=50)
    assertmsg = "Comparing nfw with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(nfw, scf, assertmsg, eps=1e-4)


## Tests whether scf Rforce matches with Hernquist Rforce
def test_RforceMatches_hernquist():
    h = potential.HernquistPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 1)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing the radial force of Hernquist Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(h.Rforce, scf.Rforce, assertmsg)


## Tests whether scf zforce matches with Hernquist zforce
def test_zforceMatches_hernquist():
    h = potential.HernquistPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 1)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing the vertical force of Hernquist Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(h.zforce, scf.zforce, assertmsg)


## Tests whether scf phitorque matches with Hernquist phitorque
def test_phitorqueMatches_hernquist():
    h = potential.HernquistPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(sphericalHernquistDensity, 1)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing the azimuth force of Hernquist Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(h.phitorque, scf.phitorque, assertmsg)


## Tests whether scf Rforce matches with NFW Rforce
def test_RforceMatches_nfw():
    nfw = potential.NFWPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW, 50, a=50)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin, a=50)
    assertmsg = "Comparing the radial force of NFW Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(nfw.Rforce, scf.Rforce, assertmsg, eps=1e-3)


## Tests whether scf zforce matches with NFW zforce
def test_zforceMatches_nfw():
    nfw = potential.NFWPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW, 50, a=50)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin, a=50)
    assertmsg = "Comparing the vertical force of NFW Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(nfw.zforce, scf.zforce, assertmsg, eps=1e-3)


## Tests whether scf phitorque matches with NFW Rforce
def test_phitorqueMatches_nfw():
    nfw = potential.NFWPotential()
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_NFW, 10)
    scf = SCFPotential(amp=1, Acos=Acos, Asin=Asin)
    assertmsg = "Comparing the azimuth force of NFW Potential with SCF fails at R={0}, Z={1}, phi={2}"
    compareFunctions(nfw.phitorque, scf.phitorque, assertmsg)


# Test that "FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated ..." warning doesn't happen (#461)
def test_FutureWarning_multid_indexing():
    scf = SCFPotential()
    array = numpy.linspace(0, 3, 100)
    # Turn warnings into errors to test for them
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", FutureWarning)
        ArrayTest(scf, [array, 1.0, 0])
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                "Using a non-tuple sequence for multidimensional indexing is deprecated"
                in str(wa.message)
            )
            if raisedWarning:
                break
        assert not raisedWarning, (
            "SCFPotential should not raise 'FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated ...', but did"
        )
    return None


# Test that running with a density in physical units works as expected
def test_physical_dens_spherical():
    a = 1.3
    ro, vo = 7.0, 230.0
    hp = potential.HernquistPotential(a=a, ro=ro, vo=vo)
    Acos, Asin = potential.scf_compute_coeffs_spherical(hp.dens, 10, a=a)
    sp = potential.SCFPotential(Acos=Acos, Asin=Asin, a=a)
    rs = numpy.geomspace(0.1, 10.0, 101)
    assert numpy.all(
        numpy.fabs(
            1.0
            - sp.dens(rs, 0.0, use_physical=False)
            / hp.dens(rs, 0.0, use_physical=False)
        )
        < 1e-10
    ), (
        "SCF density does not agree with input density when calculated with physical density"
    )
    return None


# Test that running with a density in physical units works as expected
def test_physical_dens_axi():
    a = 1.3
    ro, vo = 7.0, 230.0
    hp = potential.HernquistPotential(a=a, ro=ro, vo=vo)
    Acos, Asin = potential.scf_compute_coeffs_axi(hp.dens, 10, 2, a=a)
    sp = potential.SCFPotential(Acos=Acos, Asin=Asin, a=a)
    rs = numpy.geomspace(0.1, 10.0, 101)
    assert numpy.all(
        numpy.fabs(
            1.0
            - sp.dens(rs, 0.0, use_physical=False)
            / hp.dens(rs, 0.0, use_physical=False)
        )
        < 1e-10
    ), (
        "SCF density does not agree with input density when calculated with physical density"
    )
    return None


# Test that running with a density in physical units works as expected
def test_physical_dens():
    a = 1.3
    ro, vo = 7.0, 230.0
    hp = potential.HernquistPotential(a=a, ro=ro, vo=vo)
    Acos, Asin = potential.scf_compute_coeffs(hp.dens, 10, 2, a=a)
    sp = potential.SCFPotential(Acos=Acos, Asin=Asin, a=a)
    rs = numpy.geomspace(0.1, 10.0, 101)
    assert numpy.all(
        numpy.fabs(
            1.0
            - sp.dens(rs, 0.0, use_physical=False)
            / hp.dens(rs, 0.0, use_physical=False)
        )
        < 1e-10
    ), (
        "SCF density does not agree with input density when calculated with physical density"
    )
    return None


# Test that from_density acts as expected
def test_from_density_hernquist():
    a = 1.3
    hp = potential.HernquistPotential(a=a)
    Acos, Asin = potential.scf_compute_coeffs_spherical(hp.dens, 10, a=a)
    sp_direct = potential.SCFPotential(Acos=Acos, Asin=Asin, a=a)
    sp_from = potential.SCFPotential.from_density(
        hp.dens, 10, a=a, symmetry="spherical"
    )
    rs = numpy.geomspace(0.1, 10.0, 101)
    assert numpy.all(
        numpy.fabs(
            1.0
            - sp_direct.dens(rs, 0.0, use_physical=False)
            / sp_from.dens(rs, 0.0, use_physical=False)
        )
        < 1e-10
    ), "SCF density does not agree between direct init and from_density init"
    return None


# Test that from_density acts as expected
def test_from_density_axi():
    a = 1.0
    Acos, Asin = potential.scf_compute_coeffs_axi(
        axi_density2, 10, 10, a=a, radial_order=30, costheta_order=12
    )
    sp_direct = potential.SCFPotential(Acos=Acos, Asin=Asin, a=a)
    sp_from = potential.SCFPotential.from_density(
        axi_density2, 10, L=10, a=a, symmetry="axi", radial_order=30, costheta_order=12
    )
    rs = numpy.geomspace(0.1, 10.0, 101)
    assert numpy.all(
        numpy.fabs(
            1.0
            - sp_direct.dens(rs, rs, use_physical=False)
            / sp_from.dens(rs, rs, use_physical=False)
        )
        < 1e-10
    ), "SCF density does not agree between direct init and from_density init"
    return None


# Test that from_density acts as expected
def test_from_density():
    a = 1.0
    Acos, Asin = potential.scf_compute_coeffs(rho_Zeeuw, 10, 3, a=a)
    sp_direct = potential.SCFPotential(Acos=Acos, Asin=Asin, a=a)
    sp_from = potential.SCFPotential.from_density(
        rho_Zeeuw, 10, L=3, a=a, symmetry=None
    )
    rs = numpy.geomspace(0.1, 10.0, 101)
    assert numpy.all(
        numpy.fabs(
            1.0
            - sp_direct.dens(rs, rs, phi=rs, use_physical=False)
            / sp_from.dens(rs, rs, phi=rs, use_physical=False)
        )
        < 1e-10
    ), "SCF density does not agree between direct init and from_density init"
    return None


##############GENERIC FUNCTIONS BELOW###############


##This is used to test whether input as arrays works
def ArrayTest(scf, params):
    def compareFunctions(func, result, i):
        if numpy.isnan(result[i]):
            return numpy.isnan(func(R[i], z[i], phi[i]))
        if numpy.isinf(result[i]):
            return numpy.isinf(func(R[i], z[i], phi[i]))
        return numpy.all(numpy.fabs(result[i] - func(R[i], z[i], phi[i])) < EPS)

    potential = scf(*params).flatten()
    density = scf.dens(*params).flatten()
    Rforce = scf.Rforce(*params).flatten()
    zforce = scf.zforce(*params).flatten()
    phitorque = scf.phitorque(*params).flatten()

    R, z, phi = params
    shape = numpy.array(R * z * phi).shape
    R = (numpy.ones(shape) * R).flatten()
    z = (numpy.ones(shape) * z).flatten()
    phi = (numpy.ones(shape) * phi).flatten()
    message = "{0} at R={1}, z={2}, phi={3} was found to be {4} where it was expected to be equal to {5}"
    for i in range(len(R)):
        assert compareFunctions(scf, potential, i), message.format(
            "Potential", R[i], z[i], phi[i], potential[i], scf(R[i], z[i], phi[i])
        )
        assert compareFunctions(scf.dens, density, i), message.format(
            "Density", R[i], z[i], phi[i], density[i], scf.dens(R[i], z[i], phi[i])
        )
        assert compareFunctions(scf.Rforce, Rforce, i), message.format(
            "Rforce", R[i], z[i], phi[i], Rforce[i], scf.Rforce(R[i], z[i], phi[i])
        )
        assert compareFunctions(scf.zforce, zforce, i), message.format(
            "zforce", R[i], z[i], phi[i], zforce[i], scf.zforce(R[i], z[i], phi[i])
        )
        assert compareFunctions(scf.phitorque, phitorque, i), message.format(
            "phitorque",
            R[i],
            z[i],
            phi[i],
            phitorque[i],
            scf.phitorque(R[i], z[i], phi[i]),
        )


## This is used to compare scf functions with its corresponding galpy function
def compareFunctions(
    galpyFunc, scfFunc, assertmsg, Rs=DEFAULT_R, Zs=DEFAULT_Z, phis=DEFAULT_PHI, eps=EPS
):
    ##Assert msg must have 3 placeholders ({}) for Rs, Zs, and phis
    for ii in range(len(Rs)):
        for jj in range(len(Zs)):
            for kk in range(len(phis)):
                e = numpy.divide(
                    galpyFunc(Rs[ii], Zs[jj], phis[kk])
                    - scfFunc(Rs[ii], Zs[jj], phis[kk]),
                    galpyFunc(Rs[ii], Zs[jj], phis[kk]),
                )
                e = numpy.fabs(numpy.fabs(e))
                if galpyFunc(Rs[ii], Zs[jj], phis[kk]) == 0:
                    continue  ## Ignoring divide by zero
                assert e < eps, assertmsg.format(Rs[ii], Zs[jj], phis[kk])


##General function that tests whether coefficients for a spherical density has the expected property
def spherical_coeffsTest(Acos, Asin, eps=EPS):
    ## We expect Asin to be zero
    assert Asin is None or numpy.all(numpy.fabs(Asin) < eps), (
        "Confirming Asin = 0 fails"
    )
    ## We expect that the only non-zero values occur at (n,l=0,m=0)
    assert numpy.all(numpy.fabs(Acos[:, 1:, :]) < eps) and numpy.all(
        numpy.fabs(Acos[:, :, 1:]) < eps
    ), "Non Zero value found outside (n,l,m) = (n,0,0)"


##General function that tests whether coefficients for an axi symmetric density has the expected property
def axi_coeffsTest(Acos, Asin):
    ## We expect Asin to be zero
    assert Asin is None or numpy.all(numpy.fabs(Asin) < EPS), (
        "Confirming Asin = 0 fails"
    )
    ## We expect that the only non-zero values occur at (n,l,m=0)
    assert numpy.all(numpy.fabs(Acos[:, :, 1:]) < EPS), (
        "Non Zero value found outside (n,l,m) = (n,0,0)"
    )


## Tests whether the coefficients of a spherical density computed using the scf_compute_coeffs_axi reduces to
## The coefficients computed using the scf_compute_coeffs_spherical
def axi_reducesto_spherical(Aspherical, Aaxi, potentialName):
    Acos_s, Asin_s = Aspherical
    Acos_a, Asin_a = Aaxi

    spherical_coeffsTest(Acos_a, Asin_a, eps=1e-10)
    n = min(Acos_s.shape[0], Acos_a.shape[0])
    assert numpy.all(numpy.fabs(Acos_s[:n, 0, 0] - Acos_a[:n, 0, 0]) < EPS), (
        f"The axi symmetric Acos(n,l=0,m=0) does not reduce to the spherical Acos(n,l=0,m=0) for {potentialName}"
    )


## Tests whether the coefficients of a spherical density computed using the scf_compute_coeffs reduces to
## The coefficients computed using the scf_compute_coeffs_spherical
def reducesto_spherical(Aspherical, A, potentialName):
    Acos_s, Asin_s = Aspherical
    Acos, Asin = A

    spherical_coeffsTest(Acos, Asin, eps=1e-10)
    n = min(Acos_s.shape[0], Acos.shape[0])
    assert numpy.all(numpy.fabs(Acos_s[:n, 0, 0] - Acos[:n, 0, 0]) < EPS), (
        f"Acos(n,l=0,m=0) as generated by scf_compute_coeffs does not reduce to the spherical Acos(n,l=0,m=0) for {potentialName}"
    )


## Hernquist potential as a function of r
def sphericalHernquistDensity(R, z=0, phi=0):
    h = potential.HernquistPotential()
    return h.dens(R, z, phi)


def rho_Zeeuw(R, z, phi, a=1.0):
    r, theta, phi = coords.cyl_to_spher(R, z, phi)
    return 3.0 / (4 * numpy.pi) * numpy.power((a + r), -4.0) * a


def rho_NFW(R, z=0, phi=0.0):
    nfw = potential.NFWPotential()
    return nfw.dens(R, z, phi)


def axi_density1(R, z=0, phi=0.0):
    r, theta, phi = coords.cyl_to_spher(R, z, phi)
    h = potential.HernquistPotential()
    return h.dens(R, z, phi) * (1 + numpy.cos(theta) + numpy.cos(theta) ** 2.0)


def axi_density2(R, z=0, phi=0.0):
    spherical_coords = coords.cyl_to_spher(R, z, phi)
    theta = spherical_coords[1]
    return rho_Zeeuw(R, z, phi) * (1 + numpy.cos(theta) + numpy.cos(theta) ** 2)


def density1(R, z=0, phi=0.0):
    r, theta, phi = coords.cyl_to_spher(R, z, phi)
    h = potential.HernquistPotential(2)
    return (
        h.dens(R, z, phi)
        * (1 + numpy.cos(theta) + numpy.cos(theta) ** 2.0)
        * (1 + numpy.cos(phi) + numpy.sin(phi))
    )


# ======================================================================
# Time-dependent SCFPotential tests
# ======================================================================
#
# Time dependence lets each expansion coefficient A_nlm (cos and sin) be a
# function of time; the coefficients are sampled on a tgrid and interpolated in
# time with a cubic spline (in both Python and C). For time dependence that the
# cubic spline reproduces exactly (linear/cubic in t, or evaluated at grid
# nodes) the time-dependent potential is machine-precision-identical to a static
# potential built from the coefficients at that time, which is what most of
# these tests exercise.

_TDEP_A = 1.3


def _tdep_spherical_acos(N=8):
    hp = potential.HernquistPotential(a=_TDEP_A)
    Acos, _ = potential.scf_compute_coeffs_spherical(hp.dens, N, a=_TDEP_A)
    return Acos


def _tdep_nonaxi_coeffs(N=8, L=3):
    hp = potential.HernquistPotential(a=_TDEP_A)
    dens = lambda R, z, phi: (
        hp.dens(R, z, use_physical=False) * (1.0 + 0.15 * numpy.cos(phi))
    )
    return potential.scf_compute_coeffs(dens, N, L, a=_TDEP_A)


def _make_tdep_spherical(scale=None, tgrid=None, callable_input=True):
    if scale is None:
        scale = lambda t: 1.0 + 0.05 * t  # linear-in-t: cubic spline is exact
    Acos0 = _tdep_spherical_acos()
    if tgrid is None:
        tgrid = numpy.linspace(0.0, 5.0, 26)
    if callable_input:
        return SCFPotential(Acos=lambda t: Acos0 * scale(t), a=_TDEP_A, tgrid=tgrid)
    arr = numpy.array([Acos0 * scale(t) for t in tgrid])
    return SCFPotential(Acos=arr, a=_TDEP_A, tgrid=tgrid)


def _make_tdep_nonaxi(scale=None, tgrid=None):
    if scale is None:
        scale = lambda t: 1.0 + 0.04 * t
    Ac, As = _tdep_nonaxi_coeffs()
    if tgrid is None:
        tgrid = numpy.linspace(0.0, 4.0, 21)
    Aca = numpy.array([Ac * scale(t) for t in tgrid])
    Asa = numpy.array([As * scale(t) for t in tgrid])
    return SCFPotential(Acos=Aca, Asin=Asa, a=_TDEP_A, tgrid=tgrid)


def test_tdep_spherical_callable_matches_static():
    # Time-dependent spherical SCF (callable Acos, linear-in-t) should match a
    # static SCF built from the coefficients at t0 to machine precision.
    scale = lambda t: 1.0 + 0.05 * t
    sp = _make_tdep_spherical(scale=scale, callable_input=True)
    assert sp._tdep is True
    assert sp.isNonAxi is False
    Acos0 = _tdep_spherical_acos()
    t0 = 2.3  # not a grid node, but linear-in-t so interpolation is exact
    static = SCFPotential(Acos=Acos0 * scale(t0), a=_TDEP_A)
    mx = 0.0
    for R, z in [(1.0, 0.2), (0.5, 0.0), (2.0, 1.0), (0.3, -0.4)]:
        for meth in ["__call__", "Rforce", "zforce", "dens"]:
            f_td = sp if meth == "__call__" else getattr(sp, meth)
            f_st = static if meth == "__call__" else getattr(static, meth)
            a1 = f_td(R, z, t=t0, use_physical=False)
            a2 = f_st(R, z, use_physical=False)
            mx = max(mx, numpy.fabs(a1 - a2))
    assert mx < 1e-10, f"time-dep spherical does not match static: {mx}"


def test_tdep_spherical_array_matches_static():
    # Same but with a precomputed (Nt,N,L,M) array as input.
    scale = lambda t: 1.0 + 0.05 * t
    sp = _make_tdep_spherical(scale=scale, callable_input=False)
    assert sp._tdep is True
    Acos0 = _tdep_spherical_acos()
    t0 = 3.7
    static = SCFPotential(Acos=Acos0 * scale(t0), a=_TDEP_A)
    mx = 0.0
    for R, z in [(1.0, 0.2), (0.8, -0.3), (2.0, 1.0)]:
        mx = max(
            mx,
            numpy.fabs(
                sp(R, z, t=t0, use_physical=False) - static(R, z, use_physical=False)
            ),
        )
    assert mx < 1e-10, f"array-input time-dep spherical does not match static: {mx}"


def test_tdep_nonaxi_all_methods_match_static():
    # Non-axisymmetric time-dependent SCF: check every evaluation method matches
    # the corresponding static potential at t0.
    scale = lambda t: 1.0 + 0.04 * t
    sp = _make_tdep_nonaxi(scale=scale)
    assert sp.isNonAxi is True
    Ac, As = _tdep_nonaxi_coeffs()
    t0 = 3.1
    static = SCFPotential(Acos=Ac * scale(t0), Asin=As * scale(t0), a=_TDEP_A)
    methods = [
        "Rforce",
        "zforce",
        "phitorque",
        "dens",
        "R2deriv",
        "z2deriv",
        "phi2deriv",
        "Rzderiv",
        "Rphideriv",
        "phizderiv",
    ]
    mx = 0.0
    for R, z, phi in [(1.0, 0.2, 0.7), (0.6, 0.1, 2.0), (1.5, 0.5, 1.0)]:
        mx = max(
            mx,
            numpy.fabs(
                sp(R, z, phi=phi, t=t0, use_physical=False)
                - static(R, z, phi=phi, use_physical=False)
            ),
        )
        for meth in methods:
            a1 = getattr(sp, meth)(R, z, phi=phi, t=t0, use_physical=False)
            a2 = getattr(static, meth)(R, z, phi=phi, use_physical=False)
            mx = max(mx, numpy.fabs(a1 - a2))
    assert mx < 1e-9, f"time-dep non-axi does not match static: {mx}"


def test_tdep_mass_matches_static():
    # The enclosed-mass helper should use the time-interpolated coefficients.
    scale = lambda t: 1.0 + 0.05 * t
    sp = _make_tdep_spherical(scale=scale, callable_input=True)
    Acos0 = _tdep_spherical_acos()
    t0 = 1.7
    static = SCFPotential(Acos=Acos0 * scale(t0), a=_TDEP_A)
    for R in [0.5, 1.0, 2.5]:
        m1 = sp.mass(R, t=t0, use_physical=False)
        m2 = static.mass(R, use_physical=False)
        assert numpy.fabs(m1 - m2) < 1e-10 * numpy.fabs(m2) + 1e-12


def test_tdep_reduces_to_static():
    # Constant-in-time coefficients passed with a tgrid should match the static
    # potential at any time.
    Acos0 = _tdep_spherical_acos()
    tgrid = numpy.linspace(0.0, 5.0, 11)
    sp = SCFPotential(Acos=lambda t: Acos0, a=_TDEP_A, tgrid=tgrid)
    static = SCFPotential(Acos=Acos0, a=_TDEP_A)
    for t in [0.0, 1.3, 4.9]:
        for R, z in [(1.0, 0.2), (0.5, -0.3)]:
            assert (
                numpy.fabs(
                    sp(R, z, t=t, use_physical=False) - static(R, z, use_physical=False)
                )
                < 1e-12
            )


def test_tdep_grid_node_exact():
    # At a grid node the cubic-spline interpolation is exact for ANY time
    # dependence, so a non-polynomial (sin) scaling matches static there.
    scale = lambda t: 1.0 + 0.1 * numpy.sin(t)
    tgrid = numpy.linspace(0.0, 10.0, 21)
    sp = _make_tdep_spherical(scale=scale, tgrid=tgrid, callable_input=True)
    Acos0 = _tdep_spherical_acos()
    tn = tgrid[7]
    static = SCFPotential(Acos=Acos0 * scale(tn), a=_TDEP_A)
    assert (
        numpy.fabs(
            sp(1.0, 0.3, t=tn, use_physical=False)
            - static(1.0, 0.3, use_physical=False)
        )
        < 1e-12
    )


def test_tdep_isNonAxi_detection():
    # Axisymmetric coefficients -> isNonAxi False even with a general shape;
    # genuinely non-axisymmetric sin/m>0 terms -> isNonAxi True.
    sp_ax = _make_tdep_spherical()
    assert sp_ax.isNonAxi is False
    # axi coefficients with L>1, M>1 but no m>0 power, Asin all zero
    Ac, _ = _tdep_nonaxi_coeffs()
    Ac_axi = Ac.copy()
    Ac_axi[:, :, 1:] = 0.0  # zero out all m>0
    tgrid = numpy.linspace(0.0, 4.0, 11)
    Aca = numpy.array([Ac_axi for _ in tgrid])
    Asa = numpy.zeros_like(Aca)
    sp = SCFPotential(Acos=Aca, Asin=Asa, a=_TDEP_A, tgrid=tgrid)
    assert sp.isNonAxi is False
    sp_nonaxi = _make_tdep_nonaxi()
    assert sp_nonaxi.isNonAxi is True


def test_tdep_hasC_flags():
    sp = _make_tdep_spherical()
    assert sp.hasC
    assert sp.hasC_dxdv
    assert sp.hasC_dxdv3d
    assert sp.hasC_dens


def test_tdep_array_t_broadcast():
    # Evaluate at an array of times: should broadcast and vary with time.
    from galpy.potential import evaluatePotentials

    sp = _make_tdep_nonaxi()
    t_arr = numpy.array([0.0, 1.0, 2.5, 3.9])
    vals = evaluatePotentials(sp, 1.0, 0.5, phi=0.7, t=t_arr, use_physical=False)
    assert vals.shape == (4,)
    assert numpy.all(numpy.isfinite(vals))
    assert not numpy.all(vals == vals[0]), "potential should vary with time"


def test_tdep_amplitude_scaling():
    # Scaling the amplitude (Force.__mul__, which deep-copies) should scale the
    # potential of a time-dependent SCF (and not crash on the cubic splines).
    sp = _make_tdep_spherical()
    sp2 = 3.0 * sp
    for R, z, t in [(1.0, 0.2, 1.5), (0.7, -0.1, 3.0)]:
        v = sp(R, z, t=t, use_physical=False)
        v2 = sp2(R, z, t=t, use_physical=False)
        assert numpy.fabs(v2 - 3.0 * v) < 1e-10 * numpy.fabs(3.0 * v) + 1e-12


# ---------------------- C implementation parity ----------------------


def test_static_c_orbit_parity():
    # A static SCF orbit in C should match Python. This also exercises the
    # static (Nt=0) branch of the C argument parsing and coefficient handling,
    # which the extra time-dependent header/cache slot must not have broken.
    hp = potential.HernquistPotential(a=_TDEP_A)
    Acos, _ = potential.scf_compute_coeffs_axi(hp.dens, 8, 4, a=_TDEP_A)
    sp = SCFPotential(Acos=Acos, a=_TDEP_A)
    ts = numpy.linspace(0.0, 3.0, 101)
    init = [1.0, 0.1, 1.1, 0.05, 0.1, 0.2]
    oc = Orbit(init)
    op = Orbit(init)
    oc.integrate(ts, sp, method="dop853_c")
    op.integrate(ts, sp, method="dop853")
    assert numpy.max(numpy.fabs(oc.R(ts) - op.R(ts))) < 1e-6
    assert numpy.max(numpy.fabs(oc.z(ts) - op.z(ts))) < 1e-6


def test_tdep_c_orbit_spherical():
    # Integrate the same orbit with C and Python; for linear-in-t coefficients
    # the interpolation is identical, so trajectories agree to integrator error.
    sp = _make_tdep_spherical()
    ts = numpy.linspace(0.0, 3.0, 101)
    init = [1.0, 0.1, 1.1, 0.05, 0.1, 0.2]
    oc = Orbit(init)
    op = Orbit(init)
    oc.integrate(ts, sp, method="dop853_c")
    op.integrate(ts, sp, method="dop853")
    assert numpy.max(numpy.fabs(oc.R(ts) - op.R(ts))) < 1e-6
    assert numpy.max(numpy.fabs(oc.z(ts) - op.z(ts))) < 1e-6


def test_tdep_c_orbit_nonaxi():
    sp = _make_tdep_nonaxi()
    ts = numpy.linspace(0.0, 3.0, 101)
    init = [1.0, 0.1, 1.1, 0.1, 0.05, 0.3]
    oc = Orbit(init)
    op = Orbit(init)
    oc.integrate(ts, sp, method="dop853_c")
    op.integrate(ts, sp, method="dop853")
    assert numpy.max(numpy.fabs(oc.R(ts) - op.R(ts))) < 1e-6
    assert numpy.max(numpy.fabs(oc.phi(ts) - op.phi(ts))) < 1e-6


def test_tdep_c_full_dxdv():
    # 3D variational integration exercises the full C Hessian at each time.
    sp = _make_tdep_nonaxi()
    ts = numpy.linspace(0.0, 2.0, 51)
    init = [1.0, 0.1, 1.1, 0.1, 0.05, 0.3]
    dxdv = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    oc = Orbit(init)
    op = Orbit(init)
    oc.integrate_dxdv(dxdv, ts, sp, method="dop853_c")
    op.integrate_dxdv(dxdv, ts, sp, method="dop853")
    rc = oc.getOrbit_dxdv()
    rp = op.getOrbit_dxdv()
    assert numpy.all(numpy.isfinite(rc))
    assert numpy.max(numpy.fabs(rc - rp)) < 1e-5


def test_tdep_c_planar_dxdv():
    # Planar variational integration exercises the C planar 2nd derivatives, for
    # both the non-axisymmetric and axisymmetric branches of the C summation.
    ts = numpy.linspace(0.0, 2.0, 51)
    init = [1.0, 0.1, 1.1, 0.3]
    dxdv = [1.0, 0.0, 0.0, 0.0]
    for sp in [_make_tdep_nonaxi(), _make_tdep_spherical()]:
        oc = Orbit(init)
        op = Orbit(init)
        oc.integrate_dxdv(dxdv, ts, sp, method="dop853_c")
        op.integrate_dxdv(dxdv, ts, sp, method="dop853")
        rc = oc.getOrbit_dxdv()
        rp = op.getOrbit_dxdv()
        assert numpy.all(numpy.isfinite(rc))
        assert numpy.max(numpy.fabs(rc - rp)) < 1e-5


def test_tdep_c_direct_evaluators():
    # Direct C evaluation of the potential, forces, and 2nd derivatives at t=0
    # (via the interpolation-grid C helpers) should match Python.
    from galpy.potential.interpRZPotential import (
        eval_2ndderiv_c,
        eval_force_c,
        eval_potential_c,
    )

    R = numpy.array([0.8, 1.2, 2.0])
    z = numpy.array([0.1, 0.3, -0.2])
    for sp in [_make_tdep_spherical(), _make_tdep_nonaxi()]:
        pc = eval_potential_c(sp, R, z)[0]
        pp = numpy.array(
            [sp(rr, zz, phi=0.0, t=0.0, use_physical=False) for rr, zz in zip(R, z)]
        )
        assert numpy.max(numpy.fabs(pc - pp)) < 1e-9
        fc = eval_force_c(sp, R, z)[0]
        fp = numpy.array(
            [
                sp.Rforce(rr, zz, phi=0.0, t=0.0, use_physical=False)
                for rr, zz in zip(R, z)
            ]
        )
        assert numpy.max(numpy.fabs(fc - fp)) < 1e-8
        zc = eval_force_c(sp, R, z, zforce=True)[0]
        zp = numpy.array(
            [
                sp.zforce(rr, zz, phi=0.0, t=0.0, use_physical=False)
                for rr, zz in zip(R, z)
            ]
        )
        assert numpy.max(numpy.fabs(zc - zp)) < 1e-8
        for deriv, meth in [
            ("r2deriv", "R2deriv"),
            ("z2deriv", "z2deriv"),
            ("rzderiv", "Rzderiv"),
        ]:
            dc = eval_2ndderiv_c(sp, R, z, deriv=deriv)[0]
            dp = numpy.array(
                [
                    getattr(sp, meth)(rr, zz, phi=0.0, t=0.0, use_physical=False)
                    for rr, zz in zip(R, z)
                ]
            )
            assert numpy.max(numpy.fabs(dc - dp)) < 1e-7


def test_tdep_c_beyond_tgrid():
    # Integrating past the end of tgrid extrapolates using the boundary cubic in
    # both C and Python (clamped time-interval selection), so they still agree.
    tgrid = numpy.linspace(0.0, 2.0, 11)
    sp = _make_tdep_spherical(tgrid=tgrid)
    ts = numpy.linspace(0.0, 4.0, 101)  # extends beyond tgrid[-1]=2.0
    init = [1.0, 0.1, 1.1, 0.05, 0.1, 0.2]
    oc = Orbit(init)
    op = Orbit(init)
    oc.integrate(ts, sp, method="dop853_c")
    op.integrate(ts, sp, method="dop853")
    assert numpy.all(numpy.isfinite(oc.R(ts)))
    assert numpy.max(numpy.fabs(oc.R(ts) - op.R(ts))) < 1e-6


def test_tdep_c_dynamical_friction_dens():
    # Dynamical friction evaluates the (time-dependent) background density in C,
    # exercising the C SCFPotentialDens time-dependent path. Use an axisymmetric
    # (L>1) density so the l>0 radial-basis recursion is exercised too.
    hp = potential.HernquistPotential(a=_TDEP_A)
    Acos0, _ = potential.scf_compute_coeffs_axi(hp.dens, 8, 4, a=_TDEP_A)
    tgrid = numpy.linspace(0.0, 5.0, 26)
    sp = SCFPotential(Acos=lambda t: Acos0 * (1.0 + 0.03 * t), a=_TDEP_A, tgrid=tgrid)
    cdf = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.01, const_lnLambda=5.0, dens=sp, sigmar=lambda r: 0.7
    )
    ts = numpy.linspace(0.0, 2.0, 51)
    init = [1.0, 0.0, 1.0, 0.0, 0.1, 0.0]
    oc = Orbit(init)
    op = Orbit(init)
    oc.integrate(ts, sp + cdf, method="dop853_c")
    op.integrate(ts, sp + cdf, method="dop853")
    assert numpy.all(numpy.isfinite(oc.r(ts)))
    assert numpy.max(numpy.fabs(oc.r(ts) - op.r(ts))) < 1e-4


# ---------------------- from_density time dependence ----------------------


def test_tdep_from_density_callable_t():
    # Density with a t keyword (linear-in-t, non-axi) -> time-dependent SCF that
    # matches a static from_density at t0.
    hp = potential.HernquistPotential(a=_TDEP_A)
    dens_t = lambda R, z, phi, t=0.0: (
        hp.dens(R, z, use_physical=False)
        * (1.0 + 0.2 * numpy.cos(phi))
        * (1.0 + 0.03 * t)
    )
    tgrid = numpy.linspace(0.0, 6.0, 13)
    sp = SCFPotential.from_density(
        dens_t, 8, L=3, a=_TDEP_A, symmetry=None, tgrid=tgrid
    )
    assert sp._tdep is True
    assert sp.isNonAxi is True
    t0 = 2.0
    static = SCFPotential.from_density(
        lambda R, z, phi: dens_t(R, z, phi, t=t0), 8, L=3, a=_TDEP_A, symmetry=None
    )
    mx = 0.0
    for R, z, phi in [(1.0, 0.2, 0.7), (0.6, 0.1, 2.0)]:
        for meth in ["dens", "Rforce", "zforce", "phitorque"]:
            a1 = getattr(sp, meth)(R, z, phi=phi, t=t0, use_physical=False)
            a2 = getattr(static, meth)(R, z, phi=phi, use_physical=False)
            mx = max(mx, numpy.fabs(a1 - a2))
    assert mx < 1e-9, f"from_density time-dep does not match static: {mx}"


def test_tdep_from_density_potential_instance():
    # Passing a galpy Potential instance together with tgrid should build a
    # (here static-in-time) time-dependent SCF from its density.
    hp = potential.HernquistPotential(a=_TDEP_A)
    tgrid = numpy.linspace(0.0, 4.0, 9)
    sp = SCFPotential.from_density(hp, 8, a=_TDEP_A, symmetry="spherical", tgrid=tgrid)
    assert sp._tdep is True
    assert sp.isNonAxi is False
    static = SCFPotential.from_density(hp.dens, 8, a=_TDEP_A, symmetry="spherical")
    rs = numpy.geomspace(0.2, 5.0, 30)
    for t in [0.0, 2.5]:
        assert numpy.all(
            numpy.fabs(
                1.0
                - sp.dens(rs, 0.0, t=t, use_physical=False)
                / static.dens(rs, 0.0, use_physical=False)
            )
            < 1e-8
        )


def test_tdep_from_density_axi():
    # Axisymmetric time-dependent from_density (Asin is None path).
    a = 1.0
    tgrid = numpy.linspace(0.0, 5.0, 11)
    sp = SCFPotential.from_density(
        axi_density2,
        10,
        L=10,
        a=a,
        symmetry="axi",
        radial_order=30,
        costheta_order=12,
        tgrid=tgrid,
    )
    assert sp._tdep is True
    assert sp.isNonAxi is False
    assert numpy.all(sp._Asin_all == 0.0)
    static = SCFPotential.from_density(
        axi_density2, 10, L=10, a=a, symmetry="axi", radial_order=30, costheta_order=12
    )
    rs = numpy.geomspace(0.2, 5.0, 30)
    assert numpy.all(
        numpy.fabs(
            1.0
            - sp.dens(rs, rs, t=3.0, use_physical=False)
            / static.dens(rs, rs, use_physical=False)
        )
        < 1e-8
    )


def test_tdep_from_density_constant_no_t():
    # A density without a t argument, passed with tgrid, gives a constant-in-time
    # potential equal to the static one.
    hp = potential.HernquistPotential(a=_TDEP_A)
    tgrid = numpy.linspace(0.0, 5.0, 11)
    sp = SCFPotential.from_density(
        hp.dens, 10, a=_TDEP_A, symmetry="spherical", tgrid=tgrid
    )
    static = SCFPotential.from_density(hp.dens, 10, a=_TDEP_A, symmetry="spherical")
    rs = numpy.geomspace(0.2, 5.0, 30)
    for t in [0.0, 3.3]:
        assert numpy.all(
            numpy.fabs(
                sp.dens(rs, 0.0, t=t, use_physical=False)
                - static.dens(rs, 0.0, use_physical=False)
            )
            < 1e-12
        )


# ---------------------- error / warning handling ----------------------


def test_tdep_error_bad_ndim():
    # A 3D array (or wrong Nt) with tgrid is an error.
    with pytest.raises(RuntimeError):
        SCFPotential(Acos=numpy.ones((2, 3, 3)), tgrid=numpy.linspace(0, 1, 5))


def test_tdep_error_wrong_Nt():
    with pytest.raises(RuntimeError):
        SCFPotential(Acos=numpy.ones((3, 2, 3, 3)), tgrid=numpy.linspace(0, 1, 5))


def test_tdep_error_AsinNotNone_LnotequalM():
    with pytest.raises(RuntimeError):
        SCFPotential(
            Acos=numpy.ones((5, 2, 3, 4)),
            Asin=numpy.ones((5, 2, 3, 4)),
            tgrid=numpy.linspace(0, 1, 5),
        )


def test_tdep_error_AsinNone_LnotequalM():
    with pytest.raises(RuntimeError):
        SCFPotential(Acos=numpy.ones((5, 2, 3, 4)), tgrid=numpy.linspace(0, 1, 5))


def test_tdep_error_AsinNone_AcosNotaxisym():
    with pytest.raises(RuntimeError):
        SCFPotential(Acos=numpy.ones((5, 2, 3, 3)), tgrid=numpy.linspace(0, 1, 5))


def test_tdep_error_AsinShape_notequal_AcosShape():
    with pytest.raises(RuntimeError):
        SCFPotential(
            Acos=numpy.ones((5, 2, 3, 3)),
            Asin=numpy.ones((5, 4, 3, 3)),
            tgrid=numpy.linspace(0, 1, 5),
        )


def test_tdep_warning_not_lower_triangular():
    Acos = numpy.zeros((5, 2, 3, 3))
    Acos[:, :, 0, 1] = 1.0  # m>l element above the diagonal
    Asin = numpy.zeros((5, 2, 3, 3))
    with pytest.raises(RuntimeWarning):
        SCFPotential(Acos=Acos, Asin=Asin, tgrid=numpy.linspace(0, 1, 5))
