"""
Comprehensive test suite for RARPotential.

Tests cover initialization, force computation, potential evaluation,
rotation curves, density, composite use, and unit handling.
"""

import numpy
import pytest

from galpy.potential import (
    HernquistPotential,
    MiyamotoNagaiPotential,
    PlummerPotential,
    RARPotential,
)


# ---------------------------------------------------------------------------
# Shared baryonic potential fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def disk():
    return MiyamotoNagaiPotential(amp=1.0, a=3.0 / 8.0, b=0.28 / 8.0)


@pytest.fixture
def bulge():
    return PlummerPotential(amp=0.3, b=0.1)


@pytest.fixture
def halo():
    return HernquistPotential(amp=1.0, a=10.0)


# ===================================================================
#  RARPotential tests
# ===================================================================
class TestRARInit:
    """Test RARPotential initialization and parameter handling."""

    def test_default_init(self, disk):
        rp = RARPotential(disk)
        assert rp._method == "simple"
        assert rp._a0 > 0

    def test_methods(self, disk):
        for m in ("simple", "standard", "exp", "lfm"):
            rp = RARPotential(disk, method=m)
            assert rp._method == m

    def test_invalid_method_raises(self, disk):
        with pytest.raises(ValueError, match="method must be one of"):
            RARPotential(disk, method="invalid")

    def test_lfm_rejects_explicit_a0(self, disk):
        with pytest.raises(ValueError, match="a0 cannot be set"):
            RARPotential(disk, method="lfm", a0=1e-5)

    def test_lfm_a0_derived(self, disk):
        rp = RARPotential(disk, method="lfm")
        # a0 = c*H0/(2*pi) ~ 1.08e-10 m/s^2 in physical units
        # In natural units with default ro=8 kpc, vo=220 km/s,
        # a0_nat ~ 0.53 (acceleration per vo^2/ro)
        assert 0.1 < rp._a0 < 2.0

    def test_custom_a0(self, disk):
        rp = RARPotential(disk, method="simple", a0=1e-4)
        assert rp._a0 == pytest.approx(1e-4)

    def test_list_of_potentials(self, disk, bulge):
        with pytest.warns(DeprecationWarning, match="list of potentials"):
            rp = RARPotential([disk, bulge])
        assert len(rp._pot) == 2

    def test_repr(self, disk):
        rp = RARPotential(disk, method="standard")
        assert "standard" in repr(rp)
        assert "RARPotential" in repr(rp)


class TestRARForces:
    """Test that RAR forces are always stronger than baryonic."""

    def test_Rforce_boost(self, disk):
        """RAR forces must be at least as strong as baryonic."""
        rp = RARPotential(disk, method="simple")
        R, z = 1.0, 0.0
        F_bar = disk.Rforce(R, z, use_physical=False)
        F_rar = rp.Rforce(R, z, use_physical=False)
        # RAR boosts forces (both negative, RAR more negative)
        assert abs(F_rar) >= abs(F_bar)

    def test_zforce_boost(self, disk):
        rp = RARPotential(disk, method="simple")
        R, z = 1.0, 0.3
        F_bar = disk.zforce(R, z, use_physical=False)
        F_rar = rp.zforce(R, z, use_physical=False)
        assert abs(F_rar) >= abs(F_bar) - 1e-12

    @pytest.mark.parametrize("method", ["simple", "standard", "exp"])
    def test_deep_newtonian_limit(self, disk, method):
        """In the deep Newtonian regime (g >> a0), boost → 1."""
        # Use explicit large a0 to test that nu(y) → 1 for large y.
        # Default a0 is ~0.5 in natural units which is comparable to
        # galactic forces, so we set a0 very small to push y = g/a0 >> 1.
        rp = RARPotential(disk, method=method, a0=1e-10)
        R = 1.0
        F_bar = disk.Rforce(R, 0.0, use_physical=False)
        F_rar = rp.Rforce(R, 0.0, use_physical=False)
        # With a0 << g, boost should approach 1
        ratio = abs(F_rar / F_bar)
        assert 0.99 < ratio < 1.01

    def test_boosted_at_large_radius(self, disk):
        """At large radius (low acceleration), boost is significant."""
        rp = RARPotential(disk, method="simple")
        R = 50.0
        F_bar = disk.Rforce(R, 0.0, use_physical=False)
        F_rar = rp.Rforce(R, 0.0, use_physical=False)
        ratio = abs(F_rar / F_bar) if abs(F_bar) > 1e-20 else 1.0
        assert ratio > 1.01  # should see noticeable boost

    def test_methods_agree_qualitatively(self, disk):
        """All methods should boost forces in the same direction."""
        R, z = 3.0, 0.0
        F_bar = disk.Rforce(R, z, use_physical=False)
        for m in ("simple", "standard", "exp", "lfm"):
            rp = RARPotential(disk, method=m)
            F = rp.Rforce(R, z, use_physical=False)
            assert abs(F) >= abs(F_bar) - 1e-12


class TestRARInterpolation:
    """Test the nu(y) interpolation functions."""

    def test_simple_formula(self, disk):
        rp = RARPotential(disk, method="simple")
        assert rp._nu(1.0) == pytest.approx(numpy.sqrt(2.0))
        assert rp._nu(100.0) == pytest.approx(numpy.sqrt(1.01))

    def test_standard_formula(self, disk):
        rp = RARPotential(disk, method="standard")
        expected = 0.5 * (1 + numpy.sqrt(5))  # nu(1) = golden ratio
        assert rp._nu(1.0) == pytest.approx(expected)

    def test_exp_formula(self, disk):
        rp = RARPotential(disk, method="exp")
        # nu(y) = 1/(1 - exp(-sqrt(y))) at y=1
        expected = 1.0 / (1.0 - numpy.exp(-1.0))
        assert rp._nu(1.0) == pytest.approx(expected)

    def test_all_nu_approach_one_for_large_y(self, disk):
        """All interpolation functions → 1 as g/a0 → ∞."""
        for m in ("simple", "standard", "exp"):
            rp = RARPotential(disk, method=m)
            assert rp._nu(1e10) == pytest.approx(1.0, abs=1e-4)

    def test_simple_and_lfm_same_nu(self, disk):
        """'lfm' uses same interpolation as 'simple'."""
        rp_s = RARPotential(disk, method="simple")
        rp_l = RARPotential(disk, method="lfm")
        assert rp_s._nu(2.5) == pytest.approx(rp_l._nu(2.5))


class TestRARRotationCurve:
    """Test rotation curves from RAR-modified potential."""

    def test_vcirc_positive(self, disk):
        rp = RARPotential(disk, method="simple")
        R_arr = numpy.linspace(0.1, 5.0, 20)
        for R in R_arr:
            vc2 = -R * rp.Rforce(R, 0.0, use_physical=False)
            assert vc2 > 0

    def test_vcirc_flatter_than_baryonic(self, disk):
        """RAR rotation curve should decline more slowly at large R."""
        rp = RARPotential(disk, method="simple")
        R_mid, R_far = 2.0, 8.0
        vc_bar_mid = numpy.sqrt(-R_mid * disk.Rforce(R_mid, 0.0, use_physical=False))
        vc_bar_far = numpy.sqrt(-R_far * disk.Rforce(R_far, 0.0, use_physical=False))
        vc_rar_mid = numpy.sqrt(-R_mid * rp.Rforce(R_mid, 0.0, use_physical=False))
        vc_rar_far = numpy.sqrt(-R_far * rp.Rforce(R_far, 0.0, use_physical=False))
        decline_bar = vc_bar_far / vc_bar_mid
        decline_rar = vc_rar_far / vc_rar_mid
        # RAR curve should decline less (flatter)
        assert decline_rar > decline_bar


class TestRARPotentialEval:
    """Test potential evaluation for RAR."""

    def test_potential_finite(self, disk):
        rp = RARPotential(disk, method="simple")
        val = rp(1.0, 0.0, use_physical=False)
        assert numpy.isfinite(val)

    def test_potential_negative(self, disk):
        rp = RARPotential(disk, method="simple")
        val = rp(1.0, 0.0, use_physical=False)
        assert val < 0

    def test_potential_depends_on_method(self, disk):
        rp_s = RARPotential(disk, method="simple")
        rp_std = RARPotential(disk, method="standard")
        v_s = rp_s(1.0, 0.0, use_physical=False)
        v_std = rp_std(1.0, 0.0, use_physical=False)
        assert v_s != pytest.approx(v_std, rel=1e-6)


class TestRARDensity:
    """Test density computation for RAR."""

    def test_density_positive_midplane(self, disk):
        rp = RARPotential(disk, method="simple")
        rho = rp.dens(1.0, 0.0, use_physical=False)
        assert rho > 0

    def test_density_finite(self, disk):
        rp = RARPotential(disk, method="simple")
        rho = rp.dens(3.0, 0.5, use_physical=False)
        assert numpy.isfinite(rho)


class TestRARComposite:
    """Test RAR with composite baryonic potentials."""

    def test_disk_plus_bulge(self, disk, bulge):
        with pytest.warns(DeprecationWarning, match="list of potentials"):
            rp = RARPotential([disk, bulge])
        F = rp.Rforce(1.0, 0.0, use_physical=False)
        F_bar = disk.Rforce(1.0, 0.0, use_physical=False) + bulge.Rforce(
            1.0, 0.0, use_physical=False
        )
        assert abs(F) >= abs(F_bar)

    def test_three_component(self, disk, bulge, halo):
        with pytest.warns(DeprecationWarning, match="list of potentials"):
            rp = RARPotential([disk, bulge, halo])
        F = rp.Rforce(1.0, 0.0, use_physical=False)
        assert numpy.isfinite(F) and F < 0


class TestRARPhiTorque:
    """Test phi torque handling."""

    def test_phitorque_zero_axisymmetric(self, disk):
        """Wrapping axisymmetric potentials should give zero phi torque."""
        rp = RARPotential(disk, method="simple")
        tau = rp.phitorque(1.0, 0.0, use_physical=False)
        assert tau == 0.0

    def test_phitorque_zero_at_various_R(self, disk):
        """Phi torque should be zero everywhere for axisymmetric wrap."""
        rp = RARPotential(disk, method="simple")
        for R in [0.5, 1.0, 3.0, 10.0]:
            tau = rp.phitorque(R, 0.0, use_physical=False)
            assert tau == 0.0


# ===================================================================
#  Unit handling tests
# ===================================================================
class TestUnits:
    """Test physical unit conversion."""

    def test_rar_physical_output(self, disk):
        rp = RARPotential(disk, method="simple", ro=8.0, vo=220.0)
        F_phys = rp.Rforce(8.0, 0.0)  # in km/s/Myr
        F_nat = rp.Rforce(1.0, 0.0, use_physical=False)
        assert isinstance(F_phys, float)
        assert isinstance(F_nat, float)

    def test_rar_different_H0(self, disk):
        """Different H0 should give different a0 for lfm method."""
        rp_67 = RARPotential(disk, method="lfm", H0=67.4)
        rp_73 = RARPotential(disk, method="lfm", H0=73.0)
        assert rp_67._a0 != pytest.approx(rp_73._a0, rel=1e-3)
