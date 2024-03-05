############################TESTS ON POTENTIALS################################

import numpy

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
    assert scf(R, z, None) == scf(
        R, z, phi
    ), "The axisymmetric potential does not work at phi=None"
    assert scf.dens(R, z, None) == scf.dens(
        R, z, phi
    ), "The axisymmetric density does not work at phi=None"
    assert scf.Rforce(R, z, None) == scf.Rforce(
        R, z, phi
    ), "The axisymmetric Rforce does not work at phi=None"
    assert scf.zforce(R, z, None) == scf.zforce(
        R, z, phi
    ), "The axisymmetric zforce does not work at phi=None"
    assert scf.phitorque(R, z, None) == scf.phitorque(
        R, z, phi
    ), "The axisymmetric phitorque does not work at phi=None"


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
    assert (
        numpy.fabs(Acos[0, 0, 0] - 1.0) < EPS
    ), f"Acos(n=0,l=0,m=0) = 1 fails. Found to be Acos(n=0,l=0,m=0) = {Acos[0,0,0]}"
    assert numpy.all(numpy.fabs(Acos[1:, 0, 0]) < EPS), "Acos(n>0,l=0,m=0) = 0 fails."


## tests whether scf_compute_spherical computes the correct coefficients for Zeeuw's Potential
def test_scf_compute_spherical_zeeuw():
    Acos, Asin = potential.scf_compute_coeffs_spherical(rho_Zeeuw, 10)
    spherical_coeffsTest(Acos, Asin)
    assert (
        numpy.fabs(Acos[0, 0, 0] - 2 * 3.0 / 4) < EPS
    ), f"Acos(n=0,l=0,m=0) = 3/2 fails. Found to be Acos(n=0,l=0,m=0) = {Acos[0,0,0]}"
    assert (
        numpy.fabs(Acos[1, 0, 0] - 2 * 1.0 / 12) < EPS
    ), f"Acos(n=1,l=0,m=0) = 1/6 fails. Found to be Acos(n=0,l=0,m=0) = {Acos[0,0,0]}"
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
            ), f"Acos(n={n},l={l},0) = {numerically_calculated[n,l]}, whereas it was analytically calculated to be {analytically_calculated[n,l]}"
    # Checks that A at l != 0,1,2 are always zero
    assert numpy.all(numpy.fabs(A[0][:, 3:, 0]) < 1e-10), "Acos(n,l>2,m=0) = 0 fails."

    # Checks that A at n odd is always zero
    assert numpy.all(
        numpy.fabs(A[0][1::2, :, 0]) < 1e-10
    ), "Acos(n odd,l,m=0) = 0 fails."

    # Checks that A = 0 when n != 0 and l = 0
    assert numpy.all(
        numpy.fabs(A[0][1:, 0, 0]) < 1e-10
    ), "Acos(n > 1,l=0,m=0) = 0 fails."


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
            ), f"Acos(n={n},l={l},0) = {numerically_calculated[n,l]}, whereas it was analytically calculated to be {analytically_calculated[n,l]}"

    # Checks that A at l != 0,1,2 are always zero
    assert numpy.all(numpy.fabs(A[0][:, 3:, 0]) < 1e-10), "Acos(n,l>2,m=0) = 0 fails."

    # Checks that A = 0 when n = 2,4,..,2*n and l = 0
    assert numpy.all(
        numpy.fabs(A[0][2::2, 0, 0]) < 1e-10
    ), "Acos(n > 1,l = 0,m=0) = 0 fails."


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

    assert numpy.all(
        numpy.fabs(Acos - Acos2) < EPS
    ), "Increasing the radial order fails for scf_compute_coeffs_spherical"


##Tests radial and costheta order from scf_compute_coeffs_axi
def test_axi_density1_axiOrder():
    Acos, Asin = potential.scf_compute_coeffs_axi(axi_density1, 10, 10)
    Acos2, Asin2 = potential.scf_compute_coeffs_axi(
        axi_density1, 10, 10, radial_order=50, costheta_order=50
    )

    assert numpy.all(
        numpy.fabs(Acos - Acos2) < 1e-10
    ), "Increasing the radial and costheta order fails for scf_compute_coeffs_axi"


##Tests radial, costheta and phi order from scf_compute_coeffs
def test_density1_Order():
    Acos, Asin = potential.scf_compute_coeffs(density1, 5, 5)
    Acos2, Asin2 = potential.scf_compute_coeffs(
        density1, 5, 5, radial_order=19, costheta_order=19, phi_order=19
    )
    assert numpy.all(
        numpy.fabs(Acos - Acos2) < 1e-3
    ), "Increasing the radial, costheta, and phi order fails for Acos from scf_compute_coeffs"

    assert numpy.all(
        numpy.fabs(Asin - Asin) < EPS
    ), "Increasing the radial, costheta, and phi order fails for Asin from scf_compute_coeffs"


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
        assert not raisedWarning, "SCFPotential should not raise 'FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated ...', but did"
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
    ), "SCF density does not agree with input density when calculated with physical density"
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
    ), "SCF density does not agree with input density when calculated with physical density"
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
    ), "SCF density does not agree with input density when calculated with physical density"
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
    assert Asin is None or numpy.all(
        numpy.fabs(Asin) < eps
    ), "Confirming Asin = 0 fails"
    ## We expect that the only non-zero values occur at (n,l=0,m=0)
    assert numpy.all(numpy.fabs(Acos[:, 1:, :]) < eps) and numpy.all(
        numpy.fabs(Acos[:, :, 1:]) < eps
    ), "Non Zero value found outside (n,l,m) = (n,0,0)"


##General function that tests whether coefficients for an axi symmetric density has the expected property
def axi_coeffsTest(Acos, Asin):
    ## We expect Asin to be zero
    assert Asin is None or numpy.all(
        numpy.fabs(Asin) < EPS
    ), "Confirming Asin = 0 fails"
    ## We expect that the only non-zero values occur at (n,l,m=0)
    assert numpy.all(
        numpy.fabs(Acos[:, :, 1:]) < EPS
    ), "Non Zero value found outside (n,l,m) = (n,0,0)"


## Tests whether the coefficients of a spherical density computed using the scf_compute_coeffs_axi reduces to
## The coefficients computed using the scf_compute_coeffs_spherical
def axi_reducesto_spherical(Aspherical, Aaxi, potentialName):
    Acos_s, Asin_s = Aspherical
    Acos_a, Asin_a = Aaxi

    spherical_coeffsTest(Acos_a, Asin_a, eps=1e-10)
    n = min(Acos_s.shape[0], Acos_a.shape[0])
    assert numpy.all(
        numpy.fabs(Acos_s[:n, 0, 0] - Acos_a[:n, 0, 0]) < EPS
    ), f"The axi symmetric Acos(n,l=0,m=0) does not reduce to the spherical Acos(n,l=0,m=0) for {potentialName}"


## Tests whether the coefficients of a spherical density computed using the scf_compute_coeffs reduces to
## The coefficients computed using the scf_compute_coeffs_spherical
def reducesto_spherical(Aspherical, A, potentialName):
    Acos_s, Asin_s = Aspherical
    Acos, Asin = A

    spherical_coeffsTest(Acos, Asin, eps=1e-10)
    n = min(Acos_s.shape[0], Acos.shape[0])
    assert numpy.all(
        numpy.fabs(Acos_s[:n, 0, 0] - Acos[:n, 0, 0]) < EPS
    ), f"Acos(n,l=0,m=0) as generated by scf_compute_coeffs does not reduce to the spherical Acos(n,l=0,m=0) for {potentialName}"


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
