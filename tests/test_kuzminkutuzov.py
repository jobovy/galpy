# _____import packages_____

import numpy
import scipy

from galpy.actionAngle import actionAngleStaeckel, estimateDeltaStaeckel
from galpy.orbit import Orbit
from galpy.potential import (
    KuzminKutuzovStaeckelPotential,
    evaluateDensities,
    evaluatePotentials,
    evaluateRforces,
)
from galpy.util import coords


# Test whether circular velocity calculation works
def test_vcirc():
    # test the circular velocity of the KuzminKutuzovStaeckelPotential
    # using parameters from Batsleer & Dejonghe 1994, fig. 1-3
    # and their formula eq. (10)

    # _____model parameters______
    # surface ratios of disk and halo:
    ac_Ds = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 40.0, 40.0, 40.0]
    ac_Hs = [1.005, 1.005, 1.005, 1.01, 1.01, 1.01, 1.02, 1.02, 1.02]
    # disk contribution to total mass:
    ks = [0.05, 0.06, 0.07, 0.07, 0.1, 0.125, 0.1, 0.12, 0.125]
    # focal distance:
    Delta = 1.0

    for ii in range(9):
        ac_D = ac_Ds[ii]
        ac_H = ac_Hs[ii]
        k = ks[ii]

        # _____setup potential_____
        # first try, not normalized:
        V_D = KuzminKutuzovStaeckelPotential(
            amp=k, ac=ac_D, Delta=Delta, normalize=False
        )
        V_H = KuzminKutuzovStaeckelPotential(
            amp=(1.0 - k), ac=ac_H, Delta=Delta, normalize=False
        )
        pot = [V_D, V_H]
        # normalization according to Batsleer & Dejonghe 1994:
        V00 = evaluatePotentials(pot, 0.0, 0.0)
        # second try, normalized:
        V_D = KuzminKutuzovStaeckelPotential(
            amp=k / (-V00), ac=ac_D, Delta=Delta, normalize=False
        )
        V_H = KuzminKutuzovStaeckelPotential(
            amp=(1.0 - k) / (-V00), ac=ac_H, Delta=Delta, normalize=False
        )
        pot = [V_D, V_H]

        # _____calculate rotation curve_____
        Rs = numpy.linspace(0.0, 20.0, 100)
        z = 0.0
        vcirc_calc = numpy.sqrt(-Rs * evaluateRforces(pot, Rs, z))

        # _____vcirc by Batsleer & Dejonghe eq. (10) (with proper Jacobian)_____
        def vc2w(R):
            g_D = Delta**2 / (1.0 - ac_D**2)
            a_D = g_D - Delta**2
            g_H = Delta**2 / (1.0 - ac_H**2)
            a_H = g_H - Delta**2
            l = R**2 - a_D
            q = a_H - a_D
            termD = numpy.sqrt(l) * (numpy.sqrt(l) + numpy.sqrt(-g_D)) ** 2
            termH = numpy.sqrt(l - q) * (numpy.sqrt(l - q) + numpy.sqrt(-g_D - q)) ** 2
            return R**2 * (k / termD + (1.0 - k) / termH)

        vcirc_formula = numpy.sqrt(vc2w(Rs) / (-V00))

        assert numpy.all(numpy.fabs(vcirc_calc - vcirc_formula) < 10**-8.0), (
            "Calculated circular velocity for KuzminKutuzovStaeckelPotential "
            + "does not agree with eq. (10) (corrected by proper Jacobian) "
            + "by Batsleer & Dejonghe (1994)"
        )

    return None


# -----------------------------------------------------------------------------


# test whether the density calculation works
def test_density():
    # test the density calculation of the KuzminKutuzovStaeckelPotential
    # using parameters from Batsleer & Dejonghe 1994, tab. 2

    # _____parameters_____
    # table 2 in Batsleer & Dejonghe
    ac_D = [
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        75.0,
        75.0,
        75.0,
        75.0,
        75.0,
        75.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
        150.0,
    ]
    ac_H = [
        1.005,
        1.005,
        1.01,
        1.01,
        1.02,
        1.02,
        1.005,
        1.005,
        1.01,
        1.01,
        1.02,
        1.005,
        1.005,
        1.01,
        1.01,
        1.02,
        1.02,
        1.005,
        1.005,
        1.01,
        1.01,
        1.02,
        1.02,
        1.005,
        1.005,
        1.01,
        1.01,
        1.02,
        1.02,
        1.005,
        1.005,
        1.01,
        1.01,
        1.02,
        1.02,
    ]
    k = [
        0.05,
        0.08,
        0.075,
        0.11,
        0.105,
        0.11,
        0.05,
        0.08,
        0.075,
        0.11,
        0.11,
        0.05,
        0.07,
        0.07,
        0.125,
        0.1,
        0.125,
        0.05,
        0.065,
        0.07,
        0.125,
        0.10,
        0.125,
        0.05,
        0.065,
        0.07,
        0.125,
        0.10,
        0.125,
        0.05,
        0.065,
        0.075,
        0.125,
        0.11,
        0.125,
    ]
    Delta = [
        0.99,
        1.01,
        0.96,
        0.99,
        0.86,
        0.88,
        1.00,
        1.01,
        0.96,
        0.99,
        0.89,
        1.05,
        1.06,
        1.00,
        1.05,
        0.91,
        0.97,
        0.98,
        0.99,
        0.94,
        0.98,
        0.85,
        0.91,
        1.06,
        1.07,
        1.01,
        1.06,
        0.94,
        0.97,
        1.06,
        1.07,
        0.98,
        1.06,
        0.94,
        0.97,
    ]
    Mmin = [
        7.49,
        6.17,
        4.08,
        3.70,
        2.34,
        2.36,
        7.57,
        6.16,
        4.08,
        2.64,
        2.38,
        8.05,
        6.94,
        4.37,
        3.70,
        2.48,
        2.50,
        7.37,
        6.66,
        4.05,
        3.46,
        2.33,
        2.36,
        8.14,
        7.27,
        4.42,
        3.72,
        2.56,
        2.50,
        8.14,
        7.26,
        4.17,
        3.72,
        2.51,
        2.50,
    ]
    Mmax = [
        7.18,
        6.12,
        3.99,
        3.69,
        2.37,
        2.40,
        7.27,
        6.11,
        3.99,
        2.66,
        2.42,
        7.76,
        6.85,
        4.26,
        3.72,
        2.51,
        2.54,
        7.07,
        6.51,
        3.95,
        3.48,
        2.36,
        2.40,
        7.85,
        7.15,
        4.30,
        3.75,
        2.58,
        2.54,
        7.85,
        7.07,
        4.08,
        3.75,
        2.53,
        2.53,
    ]
    rhomin = [
        0.04,
        0.05,
        0.04,
        0.04,
        0.03,
        0.03,
        0.06,
        0.06,
        0.05,
        0.04,
        0.04,
        0.07,
        0.08,
        0.06,
        0.07,
        0.04,
        0.05,
        0.08,
        0.09,
        0.07,
        0.09,
        0.05,
        0.06,
        0.12,
        0.13,
        0.09,
        0.13,
        0.07,
        0.09,
        0.16,
        0.19,
        0.12,
        0.18,
        0.10,
        0.12,
    ]
    rhomax = [
        0.03,
        0.03,
        0.02,
        0.03,
        0.02,
        0.02,
        0.04,
        0.04,
        0.03,
        0.03,
        0.02,
        0.05,
        0.05,
        0.04,
        0.05,
        0.03,
        0.03,
        0.05,
        0.06,
        0.04,
        0.06,
        0.03,
        0.04,
        0.07,
        0.08,
        0.06,
        0.08,
        0.04,
        0.05,
        0.09,
        0.10,
        0.07,
        0.10,
        0.06,
        0.07,
    ]
    Sigmin = [
        58,
        52,
        52,
        49,
        39,
        40,
        58,
        55,
        51,
        44,
        40,
        59,
        54,
        53,
        49,
        41,
        42,
        58,
        55,
        51,
        48,
        39,
        40,
        59,
        55,
        53,
        49,
        42,
        42,
        59,
        55,
        52,
        49,
        42,
        42,
    ]
    Sigmax = [
        45,
        41,
        38,
        37,
        28,
        28,
        45,
        32,
        37,
        32,
        30,
        46,
        43,
        40,
        37,
        30,
        31,
        45,
        43,
        38,
        36,
        28,
        29,
        46,
        43,
        40,
        38,
        31,
        31,
        46,
        44,
        39,
        38,
        30,
        31,
    ]

    for ii in range(len(ac_D)):
        if ac_D[ii] == 40.0:
            continue
            # because I believe that there are typos in tab. 2 by Batsleer & Dejonghe...

        for jj in range(2):
            # _____parameters depending on solar position____
            if jj == 0:
                Rsun = 7.5
                zsun = 0.004
                GM = Mmin[ii]  # units: G = 1, M in 10^11 solar masses
                rho = rhomin[ii]
                Sig = Sigmin[ii]
            elif jj == 1:
                Rsun = 8.5
                zsun = 0.02
                GM = Mmax[ii]  # units: G = 1, M in 10^11 solar masses
                rho = rhomax[ii]
                Sig = Sigmax[ii]
            outstr = (
                "ac_D="
                + str(ac_D[ii])
                + ", ac_H="
                + str(ac_H[ii])
                + ", k="
                + str(k[ii])
                + ", Delta="
                + str(Delta[ii])
                + ", Mtot="
                + str(GM)
                + "*10^11Msun, Rsun="
                + str(Rsun)
                + "kpc, rho(Rsun,zsun)="
                + str(rho)
                + "Msun/pc^3, Sig(Rsun,z<1.1kpc)="
                + str(Sig)
                + "Msun/pc^2"
            )

            # _____setup potential_____
            amp_D = GM * k[ii]
            V_D = KuzminKutuzovStaeckelPotential(
                amp=amp_D, ac=ac_D[ii], Delta=Delta[ii], normalize=False
            )
            amp_H = GM * (1.0 - k[ii])
            V_H = KuzminKutuzovStaeckelPotential(
                amp=amp_H, ac=ac_H[ii], Delta=Delta[ii], normalize=False
            )
            pot = [V_D, V_H]

            # _____local density_____
            rho_calc = (
                evaluateDensities(pot, Rsun, zsun) * 100.0
            )  # units: [solar mass / pc^3]
            rho_calc = round(rho_calc, 2)

            # an error of 0.01 corresponds to the significant digit
            # given in the table, to which the density was rounded,
            # to be wrong by one.
            assert numpy.fabs(rho_calc - rho) <= 0.01 + 10.0**-8, (
                "Calculated density %f for KuzminKutuzovStaeckelPotential " % rho_calc
                + "with model parameters:\n"
                + outstr
                + "\n"
                + "does not agree with value from tab. 2 "
                + "by Batsleer & Dejonghe (1994)"
            )

            # _____surface density_____
            Sig_calc, err = scipy.integrate.quad(
                lambda z: (
                    evaluateDensities(pot, Rsun, z / 1000.0) * 100.0
                ),  # units: [solar mass / pc^3]
                0.0,
                1100.0,
            )  # units: pc
            Sig_calc = round(2.0 * Sig_calc)

            # an error of 1 corresponds to the significant digit
            # given in the table, to which the surface density was rounded,
            # to be wrong by one.
            assert numpy.fabs(Sig_calc - Sig) <= 1.0, (
                "Calculated surface density %f for KuzminKutuzovStaeckelPotential "
                % Sig_calc
                + "with model parameters:\n"
                + outstr
                + "\n"
                + "does not agree with value from tab. 2 "
                + "by Batsleer & Dejonghe (1994)"
            )

    return None


# -----------------------------------------------------------------------------


# test whether the orbit integration in C and Python are the same
def test_orbitIntegrationC():
    # _____initialize some KKSPot_____
    Delta = 1.0
    pot = KuzminKutuzovStaeckelPotential(ac=20.0, Delta=Delta, normalize=True)

    # _____initialize an orbit (twice)_____
    vxvv = [1.0, 0.1, 1.1, 0.0, 0.1]
    o_P = Orbit(vxvv=vxvv)
    o_C = Orbit(vxvv=vxvv)

    # _____integrate the orbit with python and C_____
    ts = numpy.linspace(0, 100, 101)
    o_P.integrate(ts, pot, method="leapfrog")  # python
    o_C.integrate(ts, pot, method="leapfrog_c")  # C

    for ii in range(5):
        exp3 = -1.7
        if ii == 0:
            Python, CC, string, exp1, exp2 = o_P.R(ts), o_C.R(ts), "R", -5.0, -10.0
        elif ii == 1:
            Python, CC, string, exp1, exp2 = o_P.z(ts), o_C.z(ts), "z", -3.25, -4.0
        elif ii == 2:
            Python, CC, string, exp1, exp2 = o_P.vR(ts), o_C.vR(ts), "vR", -3.0, -10.0
        elif ii == 3:
            Python, CC, string, exp1, exp2, exp3 = (
                o_P.vz(ts),
                o_C.vz(ts),
                "vz",
                -3.0,
                -4.0,
                -1.3,
            )
        elif ii == 4:
            Python, CC, string, exp1, exp2 = o_P.vT(ts), o_C.vT(ts), "vT", -5.0, -10.0

        rel_diff = numpy.fabs((Python - CC) / CC) < 10.0**exp1
        abs_diff = (numpy.fabs(Python - CC) < 10.0**exp2) * (
            numpy.fabs(Python) < 10.0**exp3
        )
        assert numpy.all(rel_diff + abs_diff), (
            "Orbit integration for "
            + string
            + " coordinate different in "
            + "C and Python implementation."
        )

    return None


# -----------------------------------------------------------------------------


# test whether this is really a Staeckel potential and the Delta is constant
def test_estimateDelta():
    # _____initialize some KKSPot_____
    Delta = 1.0
    pot = KuzminKutuzovStaeckelPotential(ac=20.0, Delta=Delta, normalize=True)

    # _____initialize an orbit (twice)_____
    vxvv = [1.0, 0.1, 1.1, 0.01, 0.1]
    o = Orbit(vxvv=vxvv)

    # _____integrate the orbit with C_____
    ts = numpy.linspace(0, 101, 100)
    o.integrate(ts, pot, method="leapfrog_c")

    # ____estimate Focal length Delta_____
    # for each time step individually:
    deltas_estimate = numpy.zeros(len(ts))
    for ii in range(len(ts)):
        deltas_estimate[ii] = estimateDeltaStaeckel(pot, o.R(ts[ii]), o.z(ts[ii]))

    assert numpy.all(
        numpy.fabs(deltas_estimate - Delta) < 10.0**-8
    ), "Focal length Delta estimated along the orbit is not constant."

    # for all time steps together:
    delta_estimate = estimateDeltaStaeckel(pot, o.R(ts), o.z(ts))

    assert (
        numpy.fabs(delta_estimate - Delta) < 10.0**-8
    ), "Focal length Delta estimated from the orbit is not the same as the input focal length."

    return None


# -----------------------------------------------------------------------------


# test whether this is really a Staeckel potential and the Actions are conserved along the orbit
def test_actionConservation():
    # _____initialize some KKSPot_____
    Delta = 1.0
    pot = KuzminKutuzovStaeckelPotential(ac=20.0, Delta=Delta, normalize=True)

    # _____initialize an orbit (twice)_____
    vxvv = [1.0, 0.1, 1.1, 0.01, 0.1]
    o = Orbit(vxvv=vxvv)

    # _____integrate the orbit with C_____
    ts = numpy.linspace(0, 101, 100)
    o.integrate(ts, pot, method="leapfrog_c")

    # _____Setup ActionAngle object and calculate actions (Staeckel approximation)_____
    aAS = actionAngleStaeckel(pot=pot, delta=Delta, c=True)
    jrs, lzs, jzs = aAS(o.R(ts), o.vR(ts), o.vT(ts), o.z(ts), o.vz(ts))

    assert numpy.all(
        numpy.fabs(jrs - jrs[0]) < 10.0**-8.0
    ), "Radial action is not conserved along orbit."

    assert numpy.all(
        numpy.fabs(lzs - lzs[0]) < 10.0**-8.0
    ), "Angular momentum is not conserved along orbit."

    assert numpy.all(
        numpy.fabs(jzs - jzs[0]) < 10.0**-8.0
    ), "Vertical action is not conserved along orbit."

    return None


# -----------------------------------------------------------------------------


# test coordinate transformation
def test_lambdanu_to_Rz():
    # coordinate system:
    a = 3.0
    g = 4.0
    Delta = numpy.sqrt(g - a)
    ac = numpy.sqrt(a / g)

    # _____test float input (z=0)_____
    # coordinate transformation:
    l, n = 2.0, -4.0
    R, z = coords.lambdanu_to_Rz(l, n, ac=ac, Delta=Delta)
    # true values:
    R_true = numpy.sqrt((l + a) * (n + a) / (a - g))
    z_true = numpy.sqrt((l + g) * (n + g) / (g - a))
    # test:
    assert (
        numpy.fabs(R - R_true) < 10.0**-10.0
    ), "lambdanu_to_Rz conversion did not work as expected (R)"
    assert (
        numpy.fabs(z - z_true) < 10.0**-10.0
    ), "lambdanu_to_Rz conversion did not work as expected (z)"

    # _____Also test for arrays_____
    # coordinate transformation:
    l = numpy.array([2.0, 10.0, 20.0, 0.0])
    n = numpy.array([-4.0, -3.0, -3.5, -3.5])
    R, z = coords.lambdanu_to_Rz(l, n, ac=ac, Delta=Delta)
    # true values:
    R_true = numpy.sqrt((l + a) * (n + a) / (a - g))
    z_true = numpy.sqrt((l + g) * (n + g) / (g - a))
    # test:
    rel_diff = numpy.fabs((R - R_true) / R_true) < 10.0**-8.0
    abs_diff = (numpy.fabs(R - R_true) < 10.0**-6.0) * (numpy.fabs(R_true) < 10.0**-6.0)
    assert numpy.all(
        rel_diff + abs_diff
    ), "lambdanu_to_Rz conversion did not work as expected (R array)"

    rel_diff = numpy.fabs((z - z_true) / z_true) < 10.0**-8.0
    abs_diff = (numpy.fabs(z - z_true) < 10.0**-6.0) * (numpy.fabs(z_true) < 10.0**-6.0)
    assert numpy.all(
        rel_diff + abs_diff
    ), "lambdanu_to_Rz conversion did not work as expected (z array)"
    return None


def test_Rz_to_lambdanu():
    # coordinate system:
    a = 3.0
    g = 7.0
    Delta = numpy.sqrt(g - a)
    ac = numpy.sqrt(a / g)

    # _____test float input (R=0)_____
    # true values:
    l, n = 2.0, -3.0
    # coordinate transformation:
    lt, nt = coords.Rz_to_lambdanu(
        *coords.lambdanu_to_Rz(l, n, ac=ac, Delta=Delta), ac=ac, Delta=Delta
    )
    # test:
    assert (
        numpy.fabs(lt - l) < 10.0**-10.0
    ), "Rz_to_lambdanu conversion did not work as expected (l)"
    assert (
        numpy.fabs(nt - n) < 10.0**-10.0
    ), "Rz_to_lambdanu conversion did not work as expected (n)"

    # ___Also test for arrays___
    l = numpy.array([2.0, 10.0, 20.0, 0.0])
    n = numpy.array([-7.0, -3.0, -5.0, -5.0])
    lt, nt = coords.Rz_to_lambdanu(
        *coords.lambdanu_to_Rz(l, n, ac=ac, Delta=Delta), ac=ac, Delta=Delta
    )
    assert numpy.all(
        numpy.fabs(lt - l) < 10.0**-10.0
    ), "Rz_to_lambdanu conversion did not work as expected (l array)"
    assert numpy.all(
        numpy.fabs(nt - n) < 10.0**-10.0
    ), "Rz_to_lambdanu conversion did not work as expected (n array)"
    return None


def test_Rz_to_lambdanu_r2lt0():
    # Special case that leads to r2 just below zero
    # coordinate system:
    a = 3.0
    g = 7.0
    Delta = numpy.sqrt(g - a)
    ac = numpy.sqrt(a / g)

    # _____test float input (R=0)_____
    # true values:
    l, n = 2.0, -3.0 + 10.0**-10.0
    # coordinate transformation:
    lt, nt = coords.Rz_to_lambdanu(
        *coords.lambdanu_to_Rz(l, n, ac=ac, Delta=Delta), ac=ac, Delta=Delta
    )
    # test:
    assert (
        numpy.fabs(lt - l) < 10.0**-8.0
    ), "Rz_to_lambdanu conversion did not work as expected (l)"
    assert (
        numpy.fabs(nt - n) < 10.0**-8.0
    ), "Rz_to_lambdanu conversion did not work as expected (n)"

    # ___Also test for arrays___
    l = numpy.array([2.0, 10.0, 20.0, 0.0])
    n = numpy.array([-7.0, -3.0 + 10.0**-10.0, -5.0, -5.0])
    lt, nt = coords.Rz_to_lambdanu(
        *coords.lambdanu_to_Rz(l, n, ac=ac, Delta=Delta), ac=ac, Delta=Delta
    )
    assert numpy.all(
        numpy.fabs(lt - l) < 10.0**-8.0
    ), "Rz_to_lambdanu conversion did not work as expected (l array)"
    assert numpy.all(
        numpy.fabs(nt - n) < 10.0**-8.0
    ), "Rz_to_lambdanu conversion did not work as expected (n array)"
    return None


def test_Rz_to_lambdanu_jac():
    # coordinate system:
    a = 3.0
    g = 7.0
    Delta = numpy.sqrt(g - a)
    ac = numpy.sqrt(a / g)

    # _____test float input (R=0)_____
    R, z = 1.4, 0.1
    dR = 10.0**-8.0
    dz = 10.0**-8.0
    # R derivatives
    tmp = R + dR
    dR = tmp - R
    num_deriv_lR = (
        coords.Rz_to_lambdanu(R + dR, z, ac=ac, Delta=Delta)[0]
        - coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[0]
    ) / dR
    num_deriv_nR = (
        coords.Rz_to_lambdanu(R + dR, z, ac=ac, Delta=Delta)[1]
        - coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[1]
    ) / dR
    # z derivatives
    tmp = z + dz
    dz = tmp - z
    num_deriv_lz = (
        coords.Rz_to_lambdanu(R, z + dz, ac=ac, Delta=Delta)[0]
        - coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[0]
    ) / dR
    num_deriv_nz = (
        coords.Rz_to_lambdanu(R, z + dz, ac=ac, Delta=Delta)[1]
        - coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[1]
    ) / dR
    jac = coords.Rz_to_lambdanu_jac(R, z, Delta=Delta)
    assert (
        numpy.fabs(num_deriv_lR - jac[0, 0]) < 10.0**-6.0
    ), "jacobian d((lambda,nu))/d((R,z)) fails for (dl/dR)"
    assert (
        numpy.fabs(num_deriv_nR - jac[1, 0]) < 10.0**-6.0
    ), "jacobian d((lambda,nu))/d((R,z)) fails for (dn/dR)"
    assert (
        numpy.fabs(num_deriv_lz - jac[0, 1]) < 10.0**-6.0
    ), "jacobian d((lambda,nu))/d((R,z)) fails for (dl/dz)"
    assert (
        numpy.fabs(num_deriv_nz - jac[1, 1]) < 10.0**-6.0
    ), "jacobian d((lambda,nu))/d((R,z)) fails for (dn/dz)"

    # ___Also test for arrays___
    R = numpy.arange(1, 4) * 0.5
    z = numpy.arange(1, 4) * 0.125
    dR = 10.0**-8.0
    dz = 10.0**-8.0
    # R derivatives
    tmp = R + dR
    dR = tmp - R
    num_deriv_lR = (
        coords.Rz_to_lambdanu(R + dR, z, ac=ac, Delta=Delta)[0]
        - coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[0]
    ) / dR
    num_deriv_nR = (
        coords.Rz_to_lambdanu(R + dR, z, ac=ac, Delta=Delta)[1]
        - coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[1]
    ) / dR
    # z derivatives
    tmp = z + dz
    dz = tmp - z
    num_deriv_lz = (
        coords.Rz_to_lambdanu(R, z + dz, ac=ac, Delta=Delta)[0]
        - coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[0]
    ) / dR
    num_deriv_nz = (
        coords.Rz_to_lambdanu(R, z + dz, ac=ac, Delta=Delta)[1]
        - coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[1]
    ) / dR
    jac = coords.Rz_to_lambdanu_jac(R, z, Delta=Delta)
    assert numpy.all(
        numpy.fabs(num_deriv_lR - jac[0, 0]) < 10.0**-6.0
    ), "jacobian d((lambda,nu))/d((R,z)) fails for (dl/dR)"
    assert numpy.all(
        numpy.fabs(num_deriv_nR - jac[1, 0]) < 10.0**-6.0
    ), "jacobian d((lambda,nu))/d((R,z)) fails for (dn/dR)"
    assert numpy.all(
        numpy.fabs(num_deriv_lz - jac[0, 1]) < 10.0**-6.0
    ), "jacobian d((lambda,nu))/d((R,z)) fails for (dl/dz)"
    assert numpy.all(
        numpy.fabs(num_deriv_nz - jac[1, 1]) < 10.0**-6.0
    ), "jacobian d((lambda,nu))/d((R,z)) fails for (dn/dz)"
    return None


def test_Rz_to_lambdanu_hess():
    # coordinate system:
    a = 3.0
    g = 7.0
    Delta = numpy.sqrt(g - a)
    ac = numpy.sqrt(a / g)

    # _____test float input (R=0)_____
    R, z = 1.4, 0.1
    dR = 10.0**-5.0
    dz = 10.0**-5.0
    # R derivatives
    tmp = R + dR
    dR = tmp - R
    # z derivatives
    tmp = z + dz
    dz = tmp - z
    num_deriv_llRR = (
        coords.Rz_to_lambdanu(R + dR, z, ac=ac, Delta=Delta)[0]
        - 2.0 * coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[0]
        + coords.Rz_to_lambdanu(R - dR, z, ac=ac, Delta=Delta)[0]
    ) / dR**2.0
    num_deriv_nnRR = (
        coords.Rz_to_lambdanu(R + dR, z, ac=ac, Delta=Delta)[1]
        - 2.0 * coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[1]
        + coords.Rz_to_lambdanu(R - dR, z, ac=ac, Delta=Delta)[1]
    ) / dR**2.0
    num_deriv_llRz = (
        (
            coords.Rz_to_lambdanu(R + dR, z + dz, ac=ac, Delta=Delta)[0]
            - coords.Rz_to_lambdanu(R + dR, z - dz, ac=ac, Delta=Delta)[0]
            - coords.Rz_to_lambdanu(R - dR, z + dz, ac=ac, Delta=Delta)[0]
            + coords.Rz_to_lambdanu(R - dR, z - dz, ac=ac, Delta=Delta)[0]
        )
        / dR**2.0
        / 4.0
    )
    num_deriv_llzz = (
        coords.Rz_to_lambdanu(R, z + dz, ac=ac, Delta=Delta)[0]
        - 2.0 * coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[0]
        + coords.Rz_to_lambdanu(R, z - dz, ac=ac, Delta=Delta)[0]
    ) / dz**2.0
    num_deriv_nnzz = (
        coords.Rz_to_lambdanu(R, z + dz, ac=ac, Delta=Delta)[1]
        - 2.0 * coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[1]
        + coords.Rz_to_lambdanu(R, z - dz, ac=ac, Delta=Delta)[1]
    ) / dz**2.0
    num_deriv_nnRz = (
        (
            coords.Rz_to_lambdanu(R + dR, z + dz, ac=ac, Delta=Delta)[1]
            - coords.Rz_to_lambdanu(R + dR, z - dz, ac=ac, Delta=Delta)[1]
            - coords.Rz_to_lambdanu(R - dR, z + dz, ac=ac, Delta=Delta)[1]
            + coords.Rz_to_lambdanu(R - dR, z - dz, ac=ac, Delta=Delta)[1]
        )
        / dR**2.0
        / 4.0
    )
    hess = coords.Rz_to_lambdanu_hess(R, z, Delta=Delta)
    assert (
        numpy.fabs(num_deriv_llRR - hess[0, 0, 0]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dl/dR)"
    assert (
        numpy.fabs(num_deriv_llRz - hess[0, 0, 1]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dn/dR)"
    assert (
        numpy.fabs(num_deriv_nnRR - hess[1, 0, 0]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dn/dR)"
    assert (
        numpy.fabs(num_deriv_llzz - hess[0, 1, 1]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dl/dz)"
    assert (
        numpy.fabs(num_deriv_nnRz - hess[1, 0, 1]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dn/dz)"
    assert (
        numpy.fabs(num_deriv_nnzz - hess[1, 1, 1]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dn/dz)"

    # ___Also test for arrays___
    R = numpy.arange(1, 4) * 0.5
    z = numpy.arange(1, 4) * 0.125
    # R derivatives
    tmp = R + dR
    dR = tmp - R
    # z derivatives
    tmp = z + dz
    dz = tmp - z
    dR = 10.0**-5.0
    dz = 10.0**-5.0
    num_deriv_llRR = (
        coords.Rz_to_lambdanu(R + dR, z, ac=ac, Delta=Delta)[0]
        - 2.0 * coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[0]
        + coords.Rz_to_lambdanu(R - dR, z, ac=ac, Delta=Delta)[0]
    ) / dR**2.0
    num_deriv_nnRR = (
        coords.Rz_to_lambdanu(R + dR, z, ac=ac, Delta=Delta)[1]
        - 2.0 * coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[1]
        + coords.Rz_to_lambdanu(R - dR, z, ac=ac, Delta=Delta)[1]
    ) / dR**2.0
    num_deriv_llRz = (
        (
            coords.Rz_to_lambdanu(R + dR, z + dz, ac=ac, Delta=Delta)[0]
            - coords.Rz_to_lambdanu(R + dR, z - dz, ac=ac, Delta=Delta)[0]
            - coords.Rz_to_lambdanu(R - dR, z + dz, ac=ac, Delta=Delta)[0]
            + coords.Rz_to_lambdanu(R - dR, z - dz, ac=ac, Delta=Delta)[0]
        )
        / dR**2.0
        / 4.0
    )
    num_deriv_llzz = (
        coords.Rz_to_lambdanu(R, z + dz, ac=ac, Delta=Delta)[0]
        - 2.0 * coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[0]
        + coords.Rz_to_lambdanu(R, z - dz, ac=ac, Delta=Delta)[0]
    ) / dz**2.0
    num_deriv_nnzz = (
        coords.Rz_to_lambdanu(R, z + dz, ac=ac, Delta=Delta)[1]
        - 2.0 * coords.Rz_to_lambdanu(R, z, ac=ac, Delta=Delta)[1]
        + coords.Rz_to_lambdanu(R, z - dz, ac=ac, Delta=Delta)[1]
    ) / dz**2.0
    num_deriv_nnRz = (
        (
            coords.Rz_to_lambdanu(R + dR, z + dz, ac=ac, Delta=Delta)[1]
            - coords.Rz_to_lambdanu(R + dR, z - dz, ac=ac, Delta=Delta)[1]
            - coords.Rz_to_lambdanu(R - dR, z + dz, ac=ac, Delta=Delta)[1]
            + coords.Rz_to_lambdanu(R - dR, z - dz, ac=ac, Delta=Delta)[1]
        )
        / dR**2.0
        / 4.0
    )
    hess = coords.Rz_to_lambdanu_hess(R, z, Delta=Delta)
    assert numpy.all(
        numpy.fabs(num_deriv_llRR - hess[0, 0, 0]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dl/dR)"
    assert numpy.all(
        numpy.fabs(num_deriv_llRz - hess[0, 0, 1]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dn/dR)"
    assert numpy.all(
        numpy.fabs(num_deriv_nnRR - hess[1, 0, 0]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dn/dR)"
    assert numpy.all(
        numpy.fabs(num_deriv_llzz - hess[0, 1, 1]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dl/dz)"
    assert numpy.all(
        numpy.fabs(num_deriv_nnRz - hess[1, 0, 1]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dn/dz)"
    assert numpy.all(
        numpy.fabs(num_deriv_nnzz - hess[1, 1, 1]) < 10.0**-4.0
    ), "hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2] fails for (dn/dz)"

    return None
