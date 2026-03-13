"""
Tests for LFMPotential.

Validates the Lattice Field Medium modified-gravity potential against
known limiting behaviors and consistency checks.
"""
import numpy
import pytest

from galpy import potential
from galpy.potential import (
    LFMPotential,
    PlummerPotential,
    NFWPotential,
    evaluateRforces,
    evaluatezforces,
    evaluatePotentials,
)
from galpy.util._optional_deps import _APY_LOADED


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plummer_bar():
    """A Plummer baryonic potential with known parameters."""
    return PlummerPotential(amp=1.0, b=0.5)


@pytest.fixture
def lfm_plummer(plummer_bar):
    """LFMPotential wrapping a Plummer baryonic potential."""
    return LFMPotential(plummer_bar)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_a0(self, lfm_plummer):
        """Default a0 is derived from c*H0/(2*pi) and is positive."""
        assert lfm_plummer._a0 > 0

    def test_custom_a0(self, plummer_bar):
        """Explicit a0 overrides the default."""
        lp = LFMPotential(plummer_bar, a0=0.1)
        assert lp._a0 == pytest.approx(0.1, rel=1e-10)

    def test_list_pot(self, plummer_bar):
        """Accepts a list of baryonic potentials."""
        nfw = NFWPotential(amp=2.0, a=5.0)
        lp = LFMPotential([plummer_bar, nfw])
        assert len(lp._pot) == 2

    def test_repr(self, lfm_plummer):
        """Has a reasonable repr."""
        r = repr(lfm_plummer)
        assert "LFMPotential" in r
        assert "a0=" in r


# ---------------------------------------------------------------------------
# Force evaluation
# ---------------------------------------------------------------------------


class TestForces:
    def test_newtonian_limit(self, plummer_bar):
        """When a0 is negligible, forces match the baryonic potential."""
        lp = LFMPotential(plummer_bar, a0=1e-30)
        Rs = [0.5, 1.0, 2.0]
        for R in Rs:
            fr_bar = evaluateRforces(plummer_bar, R, 0.0, use_physical=False)
            fr_lfm = evaluateRforces(lp, R, 0.0, use_physical=False)
            numpy.testing.assert_allclose(fr_lfm, fr_bar, rtol=1e-6)

    def test_boost_always_increases_force(self, plummer_bar, lfm_plummer):
        """LFM force magnitude >= baryonic force magnitude."""
        Rs = numpy.linspace(0.3, 5.0, 20)
        for R in Rs:
            fr_bar = abs(evaluateRforces(plummer_bar, R, 0.0, use_physical=False))
            fr_lfm = abs(evaluateRforces(lfm_plummer, R, 0.0, use_physical=False))
            assert fr_lfm >= fr_bar * (1.0 - 1e-10), (
                f"At R={R}: LFM force {fr_lfm} < baryonic {fr_bar}"
            )

    def test_deep_field_limit(self, plummer_bar):
        """At large radius where g_bar << a0, g_obs ~ sqrt(g_bar * a0)."""
        lp = LFMPotential(plummer_bar, a0=100.0)  # make a0 dominate
        R = 10.0
        fr_bar = evaluateRforces(plummer_bar, R, 0.0, use_physical=False)
        fr_lfm = evaluateRforces(lp, R, 0.0, use_physical=False)
        g_bar = abs(fr_bar)
        g_lfm = abs(fr_lfm)
        g_deep = numpy.sqrt(g_bar * 100.0)  # sqrt(g_bar * a0)
        numpy.testing.assert_allclose(g_lfm, g_deep, rtol=0.01)

    def test_force_direction_preserved(self, plummer_bar, lfm_plummer):
        """LFM forces point in the same direction as baryonic forces."""
        R, z = 1.0, 0.5
        fr_bar = evaluateRforces(plummer_bar, R, z, use_physical=False)
        fz_bar = evaluatezforces(plummer_bar, R, z, use_physical=False)
        fr_lfm = evaluateRforces(lfm_plummer, R, z, use_physical=False)
        fz_lfm = evaluatezforces(lfm_plummer, R, z, use_physical=False)
        # Same sign (direction)
        assert numpy.sign(fr_bar) == numpy.sign(fr_lfm)
        assert numpy.sign(fz_bar) == numpy.sign(fz_lfm)
        # Same angle
        angle_bar = numpy.arctan2(fz_bar, fr_bar)
        angle_lfm = numpy.arctan2(fz_lfm, fr_lfm)
        numpy.testing.assert_allclose(angle_bar, angle_lfm, atol=1e-10)

    def test_force_at_z_nonzero(self, plummer_bar, lfm_plummer):
        """Forces work off the midplane."""
        R, z = 1.0, 1.0
        fr = evaluateRforces(lfm_plummer, R, z, use_physical=False)
        fz = evaluatezforces(lfm_plummer, R, z, use_physical=False)
        assert numpy.isfinite(fr)
        assert numpy.isfinite(fz)
        assert fr < 0  # attractive inward
        assert fz < 0  # z>0 means force pulls down


# ---------------------------------------------------------------------------
# Rotation curve (the main scientific use case)
# ---------------------------------------------------------------------------


class TestRotationCurve:
    def test_flat_rotation_curve(self):
        """LFM produces flatter rotation curves than baryonic alone."""
        bp = PlummerPotential(amp=5.0, b=1.0)
        lp = LFMPotential(bp, a0=0.5)
        Rs = numpy.linspace(1.0, 10.0, 20)
        vc_bar = numpy.array(
            [bp.vcirc(R, use_physical=False) for R in Rs]
        )
        vc_lfm = numpy.array(
            [lp.vcirc(R, use_physical=False) for R in Rs]
        )
        # vc_lfm should be >= vc_bar everywhere
        assert numpy.all(vc_lfm >= vc_bar * 0.999)
        # At large R, vc_bar falls Keplerian; vc_lfm should fall slower
        ratio_inner = vc_lfm[0] / vc_bar[0]
        ratio_outer = vc_lfm[-1] / vc_bar[-1]
        assert ratio_outer > ratio_inner, (
            "LFM boost should be relatively larger at larger radius"
        )

    def test_rar_formula(self, plummer_bar):
        """Verify g_obs = sqrt(g_bar^2 + g_bar * a0) directly."""
        a0_val = 0.5
        lp = LFMPotential(plummer_bar, a0=a0_val)
        Rs = [0.5, 1.0, 2.0, 5.0]
        for R in Rs:
            g_bar = abs(evaluateRforces(plummer_bar, R, 0.0, use_physical=False))
            g_lfm = abs(evaluateRforces(lp, R, 0.0, use_physical=False))
            g_expected = numpy.sqrt(g_bar**2 + g_bar * a0_val)
            numpy.testing.assert_allclose(
                g_lfm, g_expected, rtol=1e-10,
                err_msg=f"RAR formula fails at R={R}",
            )


# ---------------------------------------------------------------------------
# Potential evaluation
# ---------------------------------------------------------------------------


class TestPotentialEvaluation:
    def test_potential_finite(self, lfm_plummer):
        """Potential returns finite values at typical positions."""
        Rs = [0.5, 1.0, 2.0]
        for R in Rs:
            phi = evaluatePotentials(lfm_plummer, R, 0.0, use_physical=False)
            assert numpy.isfinite(phi)

    def test_potential_decreases_inward(self, lfm_plummer):
        """Potential well deepens toward the center (more negative)."""
        phi_outer = evaluatePotentials(lfm_plummer, 5.0, 0.0, use_physical=False)
        phi_inner = evaluatePotentials(lfm_plummer, 0.5, 0.0, use_physical=False)
        assert phi_inner < phi_outer

    def test_newtonian_limit_potential(self, plummer_bar):
        """When a0 ~ 0, potential matches baryonic (up to integration cutoff)."""
        lp = LFMPotential(plummer_bar, a0=1e-30, rmax=200.0)
        R = 1.0
        phi_bar = evaluatePotentials(plummer_bar, R, 0.0, use_physical=False)
        phi_lfm = evaluatePotentials(lp, R, 0.0, use_physical=False)
        # Won't match exactly (numerical quadrature), but direction correct
        assert phi_lfm < 0
        # Should be close to baryonic
        numpy.testing.assert_allclose(phi_lfm, phi_bar, rtol=0.05)


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------


class TestDensity:
    def test_density_positive_near_center(self, lfm_plummer):
        """Effective density is positive near the center."""
        rho = lfm_plummer.dens(1.0, 0.0, use_physical=False)
        assert rho > 0

    def test_phantom_dark_matter(self, plummer_bar, lfm_plummer):
        """LFM effective density > baryonic density (phantom DM component)."""
        R = 2.0
        rho_bar = plummer_bar.dens(R, 0.0, use_physical=False)
        rho_lfm = lfm_plummer.dens(R, 0.0, use_physical=False)
        assert rho_lfm > rho_bar * 0.99


# ---------------------------------------------------------------------------
# Composite potential
# ---------------------------------------------------------------------------


class TestComposite:
    def test_nfw_plus_disk(self):
        """LFM works with a composite baryonic model (NFW + Plummer)."""
        nfw = NFWPotential(amp=2.0, a=5.0)
        plummer = PlummerPotential(amp=1.0, b=0.5)
        lp = LFMPotential([nfw, plummer], a0=0.5)
        fr = evaluateRforces(lp, 1.0, 0.0, use_physical=False)
        assert numpy.isfinite(fr)
        assert fr < 0

    def test_addable(self, lfm_plummer):
        """LFM potential can be added to other potentials."""
        nfw = NFWPotential(amp=2.0, a=5.0)
        combined = lfm_plummer + nfw
        fr = evaluateRforces(combined, 1.0, 0.0, use_physical=False)
        assert numpy.isfinite(fr)


# ---------------------------------------------------------------------------
# a0 units
# ---------------------------------------------------------------------------


class TestUnits:
    def test_a0_default_value(self, plummer_bar):
        """Default a0 in natural units is approximately 0.53 for ro=8, vo=220."""
        lp = LFMPotential(plummer_bar, ro=8.0, vo=220.0)
        numpy.testing.assert_allclose(lp._a0, 0.53, atol=0.02)

    @pytest.mark.skipif(not _APY_LOADED, reason="astropy not installed")
    def test_a0_astropy_quantity(self, plummer_bar):
        """a0 can be passed as an astropy Quantity."""
        import astropy.units as u

        lp = LFMPotential(plummer_bar, a0=1.04e-10 * u.m / u.s**2, ro=8.0, vo=220.0)
        numpy.testing.assert_allclose(lp._a0, 0.53, atol=0.02)
