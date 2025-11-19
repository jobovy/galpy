"""
Test C implementation of TwoPowerTriaxialPotential
This test verifies that the C and Python implementations give the same results
"""

import numpy as np
import pytest

from galpy import potential
from galpy.orbit import Orbit


def test_twopowertriaxial_hasC_flag():
    """Test that hasC flag is set correctly"""
    # With glorder: should have C
    tp_c = potential.TwoPowerTriaxialPotential(
        amp=1.0, a=1.5, alpha=1.5, beta=3.5, b=0.9, c=0.8, glorder=50
    )
    assert tp_c.hasC, "TwoPowerTriaxialPotential with glorder should have C"
    assert tp_c.hasC_dens, "TwoPowerTriaxialPotential should have C density"
    assert not tp_c.hasC_dxdv, (
        "TwoPowerTriaxialPotential should not have C second derivs"
    )

    # Without glorder: should not have C
    tp_py = potential.TwoPowerTriaxialPotential(
        amp=1.0, a=1.5, alpha=1.5, beta=3.5, b=0.9, c=0.8, glorder=None
    )
    assert not tp_py.hasC, "TwoPowerTriaxialPotential without glorder should not have C"


def test_twopowertriaxial_c_vs_python_potential():
    """Test that C and Python implementations give same potential values"""
    tp_c = potential.TwoPowerTriaxialPotential(
        amp=2.0, a=2.0, alpha=1.0, beta=3.0, b=0.8, c=0.7, glorder=50
    )
    tp_py = potential.TwoPowerTriaxialPotential(
        amp=2.0, a=2.0, alpha=1.0, beta=3.0, b=0.8, c=0.7, glorder=None
    )

    R = np.array([0.5, 1.0, 1.5, 2.0])
    z = np.array([0.0, 0.5, 1.0, -0.5])
    phi = np.array([0.0, 0.5, 1.0, 1.5])

    for i in range(len(R)):
        vc = potential.evaluatePotentials(tp_c, R[i], z[i], phi=phi[i])
        vpy = potential.evaluatePotentials(tp_py, R[i], z[i], phi=phi[i])
        rel_err = np.abs((vc - vpy) / vpy) if vpy != 0 else np.abs(vc - vpy)
        assert rel_err < 5e-6, (
            f"Potential mismatch at R={R[i]}, z={z[i]}, phi={phi[i]}: "
            f"C={vc}, Python={vpy}, rel_err={rel_err}"
        )


def test_twopowertriaxial_c_vs_python_forces():
    """Test that C and Python implementations give same force values"""
    tp_c = potential.TwoPowerTriaxialPotential(
        amp=2.0, a=2.0, alpha=1.5, beta=3.5, b=0.8, c=0.7, glorder=50
    )
    tp_py = potential.TwoPowerTriaxialPotential(
        amp=2.0, a=2.0, alpha=1.5, beta=3.5, b=0.8, c=0.7, glorder=None
    )

    R = np.array([0.5, 1.0, 1.5])
    z = np.array([0.0, 0.5, 1.0])
    phi = np.array([0.0, 0.5, 1.0])

    for i in range(len(R)):
        # R force
        Rc = potential.evaluateRforces(tp_c, R[i], z[i], phi=phi[i])
        Rpy = potential.evaluateRforces(tp_py, R[i], z[i], phi=phi[i])
        rel_err = np.abs((Rc - Rpy) / Rpy) if Rpy != 0 else np.abs(Rc - Rpy)
        assert rel_err < 5e-6, (
            f"Rforce mismatch: C={Rc}, Python={Rpy}, rel_err={rel_err}"
        )

        # z force
        zc = potential.evaluatezforces(tp_c, R[i], z[i], phi=phi[i])
        zpy = potential.evaluatezforces(tp_py, R[i], z[i], phi=phi[i])
        rel_err = np.abs((zc - zpy) / zpy) if zpy != 0 else np.abs(zc - zpy)
        assert rel_err < 5e-6, (
            f"zforce mismatch: C={zc}, Python={zpy}, rel_err={rel_err}"
        )

        # phi torque
        phic = potential.evaluatephitorques(tp_c, R[i], z[i], phi=phi[i])
        phipy = potential.evaluatephitorques(tp_py, R[i], z[i], phi=phi[i])
        rel_err = np.abs((phic - phipy) / phipy) if phipy != 0 else np.abs(phic - phipy)
        assert rel_err < 5e-6, (
            f"phitorque mismatch: C={phic}, Python={phipy}, rel_err={rel_err}"
        )


def test_twopowertriaxial_c_vs_python_density():
    """Test that C and Python implementations give same density values"""
    tp_c = potential.TwoPowerTriaxialPotential(
        amp=2.0, a=2.0, alpha=1.0, beta=3.0, b=0.8, c=0.7, glorder=50
    )
    tp_py = potential.TwoPowerTriaxialPotential(
        amp=2.0, a=2.0, alpha=1.0, beta=3.0, b=0.8, c=0.7, glorder=None
    )

    R = np.array([0.5, 1.0, 1.5])
    z = np.array([0.0, 0.5, 1.0])
    phi = np.array([0.0, 0.5, 1.0])

    for i in range(len(R)):
        dc = potential.evaluateDensities(tp_c, R[i], z[i], phi=phi[i])
        dpy = potential.evaluateDensities(tp_py, R[i], z[i], phi=phi[i])
        rel_err = np.abs((dc - dpy) / dpy) if dpy != 0 else np.abs(dc - dpy)
        assert rel_err < 5e-6, (
            f"Density mismatch: C={dc}, Python={dpy}, rel_err={rel_err}"
        )


def test_twopowertriaxial_special_case_twominusalpha_zero():
    """Test special case where twominusalpha == 0 (alpha=2)"""
    tp_c = potential.TwoPowerTriaxialPotential(
        amp=1.0, a=1.5, alpha=2.0, beta=3.5, b=0.9, c=0.8, glorder=50
    )
    tp_py = potential.TwoPowerTriaxialPotential(
        amp=1.0, a=1.5, alpha=2.0, beta=3.5, b=0.9, c=0.8, glorder=None
    )

    assert abs(tp_c.twominusalpha) < 1e-10

    R, z, phi = 1.0, 0.5, 0.5
    vc = potential.evaluatePotentials(tp_c, R, z, phi=phi)
    vpy = potential.evaluatePotentials(tp_py, R, z, phi=phi)
    assert np.isfinite(vc) and np.isfinite(vpy)
    # This special case may have slightly larger numerical differences
    rel_err = np.abs((vc - vpy) / vpy) if vpy != 0 else np.abs(vc - vpy)
    assert rel_err < 1e-3, (
        f"Special case mismatch: C={vc}, Python={vpy}, rel_err={rel_err}"
    )


def test_twopowertriaxial_orbit_integration_c():
    """Test orbit integration with C implementation"""
    tp = potential.TwoPowerTriaxialPotential(
        amp=1.0, a=1.5, alpha=1.5, beta=3.5, b=0.9, c=0.8, glorder=50
    )

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    times = np.linspace(0.0, 1.0, 10)
    o.integrate(times, tp, method="leapfrog")

    # Just verify integration completed
    assert len(o.x(times)) == len(times)
    assert all(np.isfinite(o.x(times)))
    assert all(np.isfinite(o.y(times)))
    assert all(np.isfinite(o.z(times)))


def test_twopowertriaxial_orbit_c_vs_python():
    """Test that orbit integration gives same results with C and Python"""
    tp_c = potential.TwoPowerTriaxialPotential(
        amp=2.0, a=2.0, alpha=1.5, beta=3.5, b=0.8, c=0.7, glorder=50
    )
    tp_py = potential.TwoPowerTriaxialPotential(
        amp=2.0, a=2.0, alpha=1.5, beta=3.5, b=0.8, c=0.7, glorder=None
    )

    o_c = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    o_py = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    times = np.linspace(0.0, 1.0, 10)

    o_c.integrate(times, tp_c, method="leapfrog")
    o_py.integrate(times, tp_py, method="leapfrog")

    # Compare final positions (use relative error)
    for coord in ["x", "y", "z"]:
        c_val = getattr(o_c, coord)(times[-1])
        py_val = getattr(o_py, coord)(times[-1])
        rel_err = (
            np.abs((c_val - py_val) / py_val) if py_val != 0 else np.abs(c_val - py_val)
        )
        assert rel_err < 1e-4, (
            f"{coord} mismatch: C={c_val}, Python={py_val}, rel_err={rel_err}"
        )


if __name__ == "__main__":
    # Run tests
    test_twopowertriaxial_hasC_flag()
    print("✓ hasC flag test passed")

    test_twopowertriaxial_c_vs_python_potential()
    print("✓ Potential C vs Python test passed")

    test_twopowertriaxial_c_vs_python_forces()
    print("✓ Forces C vs Python test passed")

    test_twopowertriaxial_c_vs_python_density()
    print("✓ Density C vs Python test passed")

    test_twopowertriaxial_special_case_twominusalpha_zero()
    print("✓ Special case test passed")

    test_twopowertriaxial_orbit_integration_c()
    print("✓ Orbit integration C test passed")

    test_twopowertriaxial_orbit_c_vs_python()
    print("✓ Orbit integration C vs Python test passed")

    print("\n✓ All tests passed!")
