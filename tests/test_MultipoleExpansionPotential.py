###############################################################################
# Tests for MultipoleExpansionPotential
###############################################################################
import numpy
import pytest

from galpy.potential import (
    HernquistPotential,
    MiyamotoNagaiPotential,
    MultipoleExpansionPotential,
    SCFPotential,
)

# Shared grids for reuse
_FINE_RGRID = numpy.geomspace(1e-3, 50, 401)
_DEFAULT_RGRID = numpy.geomspace(1e-2, 20, 201)


# --- Spherical tests (Hernquist) ---


def test_spherical_potential_matches_hernquist():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    for R in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        assert abs(mp(R, 0.0) - hp(R, 0.0)) / abs(hp(R, 0.0)) < 0.005, (
            f"Potential mismatch at R={R}"
        )


def test_spherical_potential_off_plane():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    pts = [(1.0, 0.5), (0.5, 1.0), (2.0, 1.0)]
    for R, z in pts:
        assert abs(mp(R, z) - hp(R, z)) / abs(hp(R, z)) < 0.005, (
            f"Potential mismatch at R={R}, z={z}"
        )


def test_spherical_density_matches_hernquist():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    for R in [0.1, 0.5, 1.0, 2.0, 5.0]:
        d_hp = hp.dens(R, 0.0)
        d_mp = mp.dens(R, 0.0)
        assert abs(d_mp - d_hp) / abs(d_hp) < 1e-5, (
            f"Density mismatch at R={R}: hp={d_hp}, mp={d_mp}"
        )


def test_spherical_isNonAxi_false():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    assert not mp.isNonAxi


# --- Axisymmetric tests (MiyamotoNagai) ---


def test_axisymmetric_potential_matches_mn():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential.from_density(
        dens=mn, L=16, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    pts = [(1.0, 0.0), (1.0, 0.5), (2.0, 0.1), (0.5, 0.5)]
    for R, z in pts:
        assert abs(mp(R, z) - mn(R, z)) / abs(mn(R, z)) < 0.02, (
            f"Potential mismatch at R={R}, z={z}"
        )


def test_axisymmetric_density_matches_mn_midplane():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential.from_density(
        dens=mn, L=16, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    for R in [0.5, 1.0, 2.0]:
        d_mn = mn.dens(R, 0.0)
        d_mp = mp.dens(R, 0.0)
        assert abs(d_mp - d_mn) / abs(d_mn) < 0.05, (
            f"Density mismatch at R={R}: mn={d_mn}, mp={d_mp}"
        )


def test_axisymmetric_isNonAxi_false():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential.from_density(
        dens=mn, L=16, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    assert not mp.isNonAxi


# --- SCF cross-validation ---


def test_scf_potential_cross_validation():
    Acos = numpy.zeros((3, 3, 1))
    Acos[0, 0, 0] = 1.0
    Acos[1, 0, 0] = 0.1
    Acos[0, 1, 0] = 0.05
    scf = SCFPotential(Acos=Acos, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=scf, L=6, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    pts = [(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (2.0, 1.0)]
    for R, z in pts:
        v_scf = scf(R, z)
        v_mp = mp(R, z)
        assert abs(v_mp - v_scf) / abs(v_scf) < 0.005, (
            f"Potential mismatch at R={R}, z={z}: scf={v_scf}, mp={v_mp}"
        )


def test_scf_only_l0_l1_nonzero():
    """For a density with only l=0 and l=1 harmonics, verify the multipole expansion
    has negligible l>=2 coefficients and correct l=0, l=1 radial functions."""
    Acos = numpy.zeros((3, 3, 1))
    Acos[0, 0, 0] = 1.0
    Acos[1, 0, 0] = 0.1
    Acos[0, 1, 0] = 0.05
    scf = SCFPotential(Acos=Acos, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=scf, L=6, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    # Only l=0 and l=1 should have non-negligible coefficients (evaluated via splines)
    rgrid = mp._rgrid
    max_l0 = numpy.max(numpy.abs(mp._rho_cos_splines[0][0](rgrid)))
    max_l1 = numpy.max(numpy.abs(mp._rho_cos_splines[1][0](rgrid)))
    assert max_l0 > 0, "l=0 coefficient must be non-zero"
    assert max_l1 > 0, "l=1 coefficient must be non-zero"
    for l in range(2, mp._L):
        max_lx = numpy.max(numpy.abs(mp._rho_cos_splines[l][0](rgrid)))
        assert max_lx < 1e-6 * max_l0, (
            f"l={l} coefficient should be negligible: {max_lx:.2e} vs l=0 max {max_l0:.2e}"
        )
    # Verify l=0 radial function: at z=0, P_1^0(0) = 0 so only l=0 contributes to density.
    # Therefore mp.dens(R, 0) should equal scf.dens(R, 0) to high precision.
    for R in [0.5, 1.0, 2.0]:
        d_mp = mp.dens(R, 0.0, use_physical=False)
        d_scf = scf.dens(R, 0.0, use_physical=False)
        assert abs(d_mp - d_scf) / abs(d_scf) < 1e-5, (
            f"l=0 radial function mismatch at midplane R={R}: mp={d_mp}, scf={d_scf}"
        )
    # Verify l=1 radial function: the l=1 term breaks z-symmetry (cos θ changes sign).
    # The difference dens(R,z) - dens(R,-z) isolates the odd-l (here l=1) contribution.
    for R, z in [(1.0, 1.0), (0.5, 1.0), (2.0, 0.5)]:
        diff_mp = mp.dens(R, z, use_physical=False) - mp.dens(R, -z, use_physical=False)
        diff_scf = scf.dens(R, z, use_physical=False) - scf.dens(
            R, -z, use_physical=False
        )
        assert abs(diff_scf) > 0, "SCF l=1 term should give z-asymmetric density"
        assert abs(diff_mp - diff_scf) / abs(diff_scf) < 1e-4, (
            f"l=1 radial function mismatch at R={R}, z={z}: diff_mp={diff_mp}, diff_scf={diff_scf}"
        )


def test_scf_density_cross_validation():
    Acos = numpy.zeros((3, 3, 1))
    Acos[0, 0, 0] = 1.0
    Acos[1, 0, 0] = 0.1
    Acos[0, 1, 0] = 0.05
    scf = SCFPotential(Acos=Acos, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=scf, L=6, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    pts = [(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (2.0, 0.0)]
    for R, z in pts:
        d_scf = scf.dens(R, z)
        d_mp = mp.dens(R, z)
        if abs(d_scf) > 1e-10:
            assert abs(d_mp - d_scf) / abs(d_scf) < 1e-5, (
                f"Density mismatch at R={R}, z={z}: scf={d_scf}, mp={d_mp}"
            )


# --- Density reconstruction ---


def test_spherical_density_reconstruction():
    coeff = 1.0 / (2.0 * numpy.pi)

    def dens(r):
        return coeff / r / (1 + r) ** 3

    mp = MultipoleExpansionPotential.from_density(
        dens=dens, L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID
    )
    for R in [0.1, 0.5, 1.0, 2.0, 5.0]:
        d_true = dens(R)
        d_mp = mp.dens(R, 0.0)
        assert abs(d_mp - d_true) / abs(d_true) < 1e-4, (
            f"Density reconstruction failed at R={R}: true={d_true}, mp={d_mp}"
        )


def test_axisymmetric_density_reconstruction():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential.from_density(
        dens=mn, L=16, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    for R in [0.5, 1.0, 2.0]:
        d_true = mn.dens(R, 0.0)
        d_mp = mp.dens(R, 0.0)
        assert abs(d_mp - d_true) / abs(d_true) < 0.05, (
            f"Density reconstruction failed at R={R}"
        )


# --- Normalization ---


def test_normalize_true():
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0),
        L=2,
        symmetry="spherical",
        normalize=True,
        rgrid=_FINE_RGRID,
    )
    vc = mp.vcirc(1.0, 0.0)
    assert abs(vc - 1.0) < 1e-10, f"vcirc(1,0) = {vc}, expected ~1.0"


def test_normalize_fraction():
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0),
        L=2,
        symmetry="spherical",
        normalize=0.5,
        rgrid=_FINE_RGRID,
    )
    vc = mp.vcirc(1.0, 0.0)
    assert abs(vc - numpy.sqrt(0.5)) < 1e-10, (
        f"vcirc(1,0) = {vc}, expected ~{numpy.sqrt(0.5)}"
    )


# --- isNonAxi ---


def test_spherical_is_axi():
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0), L=2, symmetry="spherical"
    )
    assert not mp.isNonAxi


def test_axisymmetric_is_axi():
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0), L=6, symmetry="axisymmetric"
    )
    assert not mp.isNonAxi


def test_general_with_axi_density_is_axi():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(dens=hp, L=4, symmetry=None)
    assert not mp.isNonAxi


# --- Density input variants ---


def test_potential_instance_input():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.005


def test_2arg_lambda_input():
    # rho = amp/(4*pi) * a / (r * (r+a)^3) for HernquistPotential(amp=2, a=1)
    coeff = 1.0 / (2.0 * numpy.pi)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z: (
            coeff / numpy.sqrt(R**2 + z**2) / (1 + numpy.sqrt(R**2 + z**2)) ** 3
        ),
        L=2,
        symmetry="spherical",
        rgrid=_FINE_RGRID,
    )
    hp = HernquistPotential(amp=2.0, a=1.0)
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.005


def test_1arg_lambda_input():
    coeff = 1.0 / (2.0 * numpy.pi)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda r: coeff / r / (1 + r) ** 3,
        L=2,
        symmetry="spherical",
        rgrid=_FINE_RGRID,
    )
    hp = HernquistPotential(amp=2.0, a=1.0)
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.005


# --- Edge cases ---


def test_r_zero():
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0),
        L=2,
        symmetry="spherical",
        rgrid=_FINE_RGRID,
    )
    val = mp(0.0, 0.0)
    assert numpy.isfinite(val)


def test_monopole_only():
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0),
        L=1,
        symmetry="spherical",
        rgrid=_FINE_RGRID,
    )
    hp = HernquistPotential(amp=2.0, a=1.0)
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.005


def test_OmegaP_zero():
    mp = MultipoleExpansionPotential()
    assert mp.OmegaP() == 0


def test_hasC():
    mp = MultipoleExpansionPotential()
    assert mp.hasC
    assert mp.hasC_dxdv
    assert mp.hasC_dens


def test_default_construction():
    """Test that MultipoleExpansionPotential() with no args produces a valid potential."""
    mp = MultipoleExpansionPotential()
    val = mp(1.0, 0.0)
    assert numpy.isfinite(val)
    # Should match default Hernquist
    hp = HernquistPotential(amp=2.0, a=1.0)
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.005


# --- Analytical force tests ---


def test_spherical_Rforce():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    for R in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        rf_mp = mp.Rforce(R, 0.0)
        rf_hp = hp.Rforce(R, 0.0)
        assert abs(rf_mp - rf_hp) / abs(rf_hp) < 5e-4, (
            f"Rforce mismatch at R={R}: mp={rf_mp}, hp={rf_hp}"
        )


def test_spherical_zforce():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    pts = [(1.0, 0.5), (0.5, 1.0), (2.0, 1.0)]
    for R, z in pts:
        zf_mp = mp.zforce(R, z)
        zf_hp = hp.zforce(R, z)
        assert abs(zf_mp - zf_hp) / abs(zf_hp) < 5e-5, (
            f"zforce mismatch at R={R}, z={z}: mp={zf_mp}, hp={zf_hp}"
        )


def test_axisymmetric_Rforce():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential.from_density(
        dens=mn, L=16, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    pts = [(1.0, 0.0), (1.0, 0.5), (2.0, 0.1), (0.5, 0.5)]
    for R, z in pts:
        rf_mp = mp.Rforce(R, z)
        rf_mn = mn.Rforce(R, z)
        assert abs(rf_mp - rf_mn) / abs(rf_mn) < 0.01, (
            f"Rforce mismatch at R={R}, z={z}: mp={rf_mp}, mn={rf_mn}"
        )


def test_axisymmetric_zforce():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential.from_density(
        dens=mn, L=16, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    pts = [(1.0, 0.5), (2.0, 0.1), (0.5, 0.5)]
    for R, z in pts:
        zf_mp = mp.zforce(R, z)
        zf_mn = mn.zforce(R, z)
        assert abs(zf_mp - zf_mn) / abs(zf_mn) < 0.05, (
            f"zforce mismatch at R={R}, z={z}: mp={zf_mp}, mn={zf_mn}"
        )


def test_phitorque_zero_for_axisymmetric():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    for R in [0.5, 1.0, 2.0]:
        pt = mp.phitorque(R, 0.0, phi=0.5)
        assert abs(pt) < 1e-10, f"phitorque not zero at R={R}: {pt}"


# --- Coverage: density input variants ---


def test_3arg_callable_density_input():
    """Test that a 3-argument callable density (R, z, phi) without units works."""
    coeff = 1.0 / (2.0 * numpy.pi)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi: (
            coeff / numpy.sqrt(R**2 + z**2) / (1 + numpy.sqrt(R**2 + z**2)) ** 3
        ),
        L=4,
        symmetry=None,
        rgrid=_DEFAULT_RGRID,
    )
    hp = HernquistPotential(amp=2.0, a=1.0)
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.005


def test_dens_phi_none():
    """Test that _dens handles phi=None for axisymmetric potential."""
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0),
        L=2,
        symmetry="spherical",
        rgrid=_DEFAULT_RGRID,
    )
    val = mp._dens(1.0, 0.0, phi=None)
    assert numpy.isfinite(val) and val > 0


def test_dens_at_infinity():
    """Test that density at r=infinity returns 0."""
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0),
        L=2,
        symmetry="spherical",
        rgrid=_DEFAULT_RGRID,
    )
    val = mp.dens(numpy.inf, 0.0, use_physical=False)
    assert val == 0.0


def test_spher_forces_at_r_zero():
    """Test that spherical force components at r=0 return 0."""
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0),
        L=2,
        symmetry="spherical",
        rgrid=_DEFAULT_RGRID,
    )
    dr, dtheta, dphi = mp._compute_spher_forces_at_point(0.0, 0.0, 0.0)
    assert dr == 0.0 and dtheta == 0.0 and dphi == 0.0


def test_spher_forces_at_infinity():
    """Test that spherical force components at r=infinity return 0."""
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0),
        L=2,
        symmetry="spherical",
        rgrid=_DEFAULT_RGRID,
    )
    dr, dtheta, dphi = mp._compute_spher_forces_at_point(numpy.inf, 0.0, 0.0)
    assert dr == 0.0 and dtheta == 0.0 and dphi == 0.0


# --- Second derivative tests ---


def test_2nd_derivs_at_r_zero():
    """Test that second derivatives at r=0 return all zeros."""
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0),
        L=2,
        symmetry="spherical",
        rgrid=_DEFAULT_RGRID,
    )
    result = mp._compute_spher_2nd_derivs_at_point(0.0, 0.0, 0.0)
    assert all(v == 0.0 for v in result)
    # Also test through the public interface (hits _cyl_2nd_deriv_at_point r=0 path)
    assert mp.R2deriv(0.0, 0.0, use_physical=False) == 0.0
    assert mp.z2deriv(0.0, 0.0, use_physical=False) == 0.0


def test_2nd_derivs_at_infinity():
    """Test that second derivatives at r=infinity return all zeros."""
    mp = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0),
        L=2,
        symmetry="spherical",
        rgrid=_DEFAULT_RGRID,
    )
    result = mp._compute_spher_2nd_derivs_at_point(numpy.inf, 0.0, 0.0)
    assert all(v == 0.0 for v in result)


def test_2nd_derivs_on_z_axis():
    """Test that second derivatives are finite on the z-axis (R=0, costheta=±1)
    where dP/d(costheta) diverges for m>0, triggering the pole clamping."""
    coeff = 1.0 / (2.0 * numpy.pi)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi: (
            coeff
            / numpy.sqrt(R**2 + z**2)
            / (1 + numpy.sqrt(R**2 + z**2)) ** 3
            * (1.0 + 0.1 * numpy.cos(2 * phi))
        ),
        L=6,
        symmetry=None,
        rgrid=_FINE_RGRID,
    )
    # R=0, z>0 => theta=0 => costheta=1
    for z in [0.5, 1.0, 2.0]:
        R2 = mp.R2deriv(0.0, z, use_physical=False)
        z2 = mp.z2deriv(0.0, z, use_physical=False)
        Rz = mp.Rzderiv(0.0, z, use_physical=False)
        assert numpy.isfinite(R2), f"R2deriv not finite at R=0, z={z}"
        assert numpy.isfinite(z2), f"z2deriv not finite at R=0, z={z}"
        assert numpy.isfinite(Rz), f"Rzderiv not finite at R=0, z={z}"
    # R=0, z<0 => theta=pi => costheta=-1
    for z in [-0.5, -1.0, -2.0]:
        R2 = mp.R2deriv(0.0, z, use_physical=False)
        z2 = mp.z2deriv(0.0, z, use_physical=False)
        assert numpy.isfinite(R2), f"R2deriv not finite at R=0, z={z}"
        assert numpy.isfinite(z2), f"z2deriv not finite at R=0, z={z}"


def test_2nd_derivs_on_z_axis_continuity():
    """Test that second derivatives on the z-axis are continuous with nearby points."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=6, symmetry=None, rgrid=_FINE_RGRID
    )
    z = 1.0
    R2_axis = mp.R2deriv(0.0, z, use_physical=False)
    R2_near = mp.R2deriv(1e-4, z, use_physical=False)
    assert abs(R2_axis - R2_near) / abs(R2_near) < 0.005, (
        f"R2deriv discontinuous at z-axis: on_axis={R2_axis}, near={R2_near}"
    )


def test_spherical_2nd_derivs_match_hernquist():
    """Test that second derivatives match Hernquist for a spherical expansion."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    pts = [(1.0, 0.5), (0.5, 1.0), (2.0, 0.1)]
    for R, z in pts:
        for name, func in [
            ("R2deriv", "R2deriv"),
            ("z2deriv", "z2deriv"),
            ("Rzderiv", "Rzderiv"),
        ]:
            val_mp = getattr(mp, func)(R, z, use_physical=False)
            val_hp = getattr(hp, func)(R, z, use_physical=False)
            assert abs(val_mp - val_hp) / abs(val_hp) < 5e-5, (
                f"{name} mismatch at R={R}, z={z}: mp={val_mp}, hp={val_hp}"
            )


def test_internal_spline_degree():
    """Test that the internal spline degree is set to 3."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    assert mp._k == 3
    assert numpy.isfinite(mp.R2deriv(1.0, 0.5, use_physical=False))


# --- Below/above grid extrapolation tests ---


def test_below_grid_potential_force_2ndderiv():
    """Test that potential, forces, and second derivatives are finite and
    well-behaved below the grid (r < rmin), covering the constant-density
    extrapolation in _eval_R_lm, _eval_dR_lm, _eval_d2R_lm."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp,
        L=4,
        symmetry="axisymmetric",
        rgrid=numpy.geomspace(2.0, 20.0, 201),
    )
    rmin = mp._rgrid[0]
    # Evaluate at r < rmin (R=1.0, z=0 gives r=1.0 < rmin=2.0)
    for R in [0.5, 1.0, 1.5]:
        val = mp(R, 0.0)
        rf = mp.Rforce(R, 0.0)
        r2 = mp.R2deriv(R, 0.0, use_physical=False)
        z2 = mp.z2deriv(R, 0.0, use_physical=False)
        assert numpy.isfinite(val), f"Potential not finite at R={R} < rmin"
        assert numpy.isfinite(rf), f"Rforce not finite at R={R} < rmin"
        assert rf < 0, f"Rforce should be attractive at R={R} < rmin"
        assert numpy.isfinite(r2), f"R2deriv not finite at R={R} < rmin"
        assert numpy.isfinite(z2), f"z2deriv not finite at R={R} < rmin"


def test_below_grid_l2_branch():
    """Test the l=2 special case in _below_grid_integrals (log formula),
    which requires L >= 3 and evaluation at r < rmin."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z: hp.dens(R, z) * (1 + 1e-8 * z**2),
        L=4,
        symmetry="axisymmetric",
        rgrid=numpy.geomspace(2.0, 20.0, 201),
    )
    # Evaluate at r < rmin; L=4 means l goes 0,1,2,3, hitting l=2
    R, z = 0.5, 0.5
    val = mp(R, z)
    rf = mp.Rforce(R, z)
    r2 = mp.R2deriv(R, z, use_physical=False)
    assert numpy.isfinite(val), "Potential not finite for l=2 below-grid"
    assert numpy.isfinite(rf), "Rforce not finite for l=2 below-grid"
    assert numpy.isfinite(r2), "R2deriv not finite for l=2 below-grid"


def test_above_grid_2nd_derivs():
    """Test that second derivatives are finite and behave as point-mass
    for r > rmax, covering the r > rmax branch of _eval_d2R_lm."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp,
        L=2,
        symmetry="spherical",
        rgrid=numpy.geomspace(0.01, 5.0, 201),
    )
    rmax = mp._rgrid[-1]
    # Evaluate at r > rmax
    for R in [6.0, 8.0, 10.0]:
        r2 = mp.R2deriv(R, 0.0, use_physical=False)
        z2 = mp.z2deriv(R, 0.0, use_physical=False)
        assert numpy.isfinite(r2), f"R2deriv not finite at R={R} > rmax"
        assert numpy.isfinite(z2), f"z2deriv not finite at R={R} > rmax"
    # Check point-mass behavior: R2deriv should decrease with distance
    r2_6 = mp.R2deriv(6.0, 0.0, use_physical=False)
    r2_10 = mp.R2deriv(10.0, 0.0, use_physical=False)
    assert abs(r2_6) > abs(r2_10), "R2deriv should decrease with distance"


def test_below_grid_density_clamped():
    """Test that density below the grid returns the value at rmin (clamped)."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp,
        L=2,
        symmetry="spherical",
        rgrid=numpy.geomspace(2.0, 20.0, 201),
    )
    rmin = mp._rgrid[0]
    d_at_rmin = mp.dens(rmin, 0.0, use_physical=False)
    d_below = mp.dens(1.0, 0.0, use_physical=False)
    assert d_below == d_at_rmin, (
        f"Density below grid should be clamped to rmin value: {d_below} != {d_at_rmin}"
    )


# --- C code coverage ---


def test_c_orbit_below_grid_l2():
    """Cover C code line 255 (l=2 below-grid log formula) via orbit integration in C.
    Orbit integration uses Rforce/zforce in C, which calls below_grid_integrals
    for r < rmin.  With L=4, l=2 is included and triggers the log-branch."""
    from galpy.orbit import Orbit

    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp,
        L=4,  # includes l=0,1,2,3; l=2 hits the log branch in below_grid_integrals
        symmetry="axisymmetric",
        rgrid=numpy.geomspace(2.0, 20.0, 201),  # rmin=2 > orbit's minimum r
    )
    # Orbit starting at R=1.0 < rmin=2.0 so forces are evaluated below the grid
    o = Orbit([1.0, 0.1, 1.0, 0.0, 0.1, 0.0])
    ts = numpy.linspace(0, 10, 101)
    o.integrate(ts, mp, method="leapfrog_c")
    assert numpy.all(numpy.isfinite(o.R(ts))), "Orbit R should be finite"
    assert numpy.all(numpy.isfinite(o.z(ts))), "Orbit z should be finite"


def test_c_planar_liouville_below_grid_d2R():
    """Cover C code lines 286-288 (d²R below-grid via EVAL_DERIV2 mode).
    A planar orbit.integrate_dxdv with a C integrator calls
    integratePlanarOrbit_dxdv in C, which calls MultipoleExpansionPotential-
    PlanarR2deriv → compute_multipole_spher_2nd_derivs → eval_radial_lm with
    EVAL_DERIV2. With rmin=2 > orbit radius the below-grid branch is hit."""
    from galpy.orbit import Orbit

    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp,
        L=4,
        symmetry="axisymmetric",
        rgrid=numpy.geomspace(2.0, 20.0, 201),  # rmin=2 > orbit's R=1.0
    )
    # Planar orbit [R, vR, vT, phi] at R=1.0 < rmin=2.0
    o = Orbit([1.0, 0.1, 1.0, 0.5])
    ts = numpy.linspace(0, 5, 51)
    o.integrate_dxdv([1.0, 0.0, 0.0, 0.0], ts, mp, method="dopr54_c")
    result = o.getOrbit_dxdv()
    assert numpy.all(numpy.isfinite(result)), "integrate_dxdv result should be finite"


# --- Direct __init__ (spline) tests ---


def test_round_trip_from_density_to_init():
    """Create via from_density, extract splines, create new instance, verify identical."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp1 = MultipoleExpansionPotential.from_density(
        dens=hp, L=4, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    mp2 = MultipoleExpansionPotential(
        rho_cos_splines=mp1._rho_cos_splines,
        rho_sin_splines=mp1._rho_sin_splines,
        rgrid=mp1._rgrid,
    )
    pts = [(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (2.0, 1.0)]
    for R, z in pts:
        assert abs(mp1(R, z) - mp2(R, z)) < 1e-14, f"Potential mismatch at R={R}, z={z}"
        assert (
            abs(mp1.dens(R, z, use_physical=False) - mp2.dens(R, z, use_physical=False))
            < 1e-14
        ), f"Density mismatch at R={R}, z={z}"
        assert abs(mp1.Rforce(R, z) - mp2.Rforce(R, z)) < 1e-14, (
            f"Rforce mismatch at R={R}, z={z}"
        )


def test_init_with_cos_splines_only():
    """Test that passing only rho_cos_splines (no rho_sin_splines) works."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp1 = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    mp2 = MultipoleExpansionPotential(
        rho_cos_splines=mp1._rho_cos_splines,
        rgrid=mp1._rgrid,
    )
    assert abs(mp1(1.0, 0.0) - mp2(1.0, 0.0)) < 1e-14


def test_init_rejects_non_spline_cos():
    """Test that passing non-spline rho_cos_splines raises TypeError or ValueError."""
    # Callables get a helpful ValueError about tgrid
    with pytest.raises(ValueError, match=r"rho_cos_splines\[0\]\[0\] appears to be"):
        MultipoleExpansionPotential(rho_cos_splines=[[lambda r: 0.0]])
    # Non-callable non-splines get TypeError
    with pytest.raises(TypeError, match=r"rho_cos_splines\[0\]\[0\] must be"):
        MultipoleExpansionPotential(rho_cos_splines=[[42]])


def test_init_rejects_non_spline_sin():
    """Test that passing non-spline rho_sin_splines raises TypeError or ValueError."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID
    )
    # Callables get a helpful ValueError about tgrid
    with pytest.raises(ValueError, match=r"rho_sin_splines\[0\]\[0\] appears to be"):
        MultipoleExpansionPotential(
            rho_cos_splines=mp._rho_cos_splines,
            rho_sin_splines=[[lambda r: 0.0]],
        )


def test_isNonAxi_auto_detection_from_splines():
    """Pass axisymmetric splines with M > 1 but negligible m > 0, verify truncation."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp_full = MultipoleExpansionPotential.from_density(
        dens=hp, L=4, symmetry=None, rgrid=_DEFAULT_RGRID
    )
    # mp_full should have detected axisymmetry and truncated to M=1
    assert not mp_full.isNonAxi
    assert mp_full._M == 1


# --- Time-dependent tests ---


def test_time_dependent_rotation_from_density():
    """Compare time-dependent multipole at time t with a freshly-built static multipole at the same rotated density."""
    omega = 1.3
    hp = HernquistPotential(amp=2.0, a=1.0)
    rgrid = numpy.geomspace(1e-3, 50, 201)
    tgrid = numpy.linspace(0, 10, 51)
    # Time-dependent density: cos(phi + omega*t)
    tdep_mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False) * (1.0 + 0.1 * numpy.cos(phi + omega * t))
        ),
        L=6,
        rgrid=rgrid,
        tgrid=tgrid,
    )
    # Compare at various (R, z, phi, t) with a freshly-built static version
    # at the same time (this tests that the time interpolation is accurate)
    test_times = [0.0, 0.2, 1.0, 5.0]
    test_spatial = [(1.0, 0.5, 0.3), (2.0, 0.1, 1.5)]
    for t in test_times:
        static_at_t = MultipoleExpansionPotential.from_density(
            dens=lambda R, z, phi, _t=t: (
                hp.dens(R, z, use_physical=False)
                * (1.0 + 0.1 * numpy.cos(phi + omega * _t))
            ),
            L=6,
            rgrid=rgrid,
        )
        for R, z, phi in test_spatial:
            val_static = static_at_t(R, z, phi=phi, use_physical=False)
            val_tdep = tdep_mp(R, z, phi=phi, t=t, use_physical=False)
            assert (
                numpy.abs(val_static - val_tdep) < 1e-10 * numpy.abs(val_static) + 1e-15
            ), (
                f"Potential mismatch at (R={R}, z={z}, phi={phi}, t={t}): {val_static} vs {val_tdep}"
            )


def test_time_dependent_rotation_forces():
    """Compare forces from time-dependent multipole vs freshly-built static at same t."""
    omega = 1.3
    hp = HernquistPotential(amp=2.0, a=1.0)
    rgrid = numpy.geomspace(1e-3, 50, 201)
    tgrid = numpy.linspace(0, 10, 51)
    tdep_mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False) * (1.0 + 0.1 * numpy.cos(phi + omega * t))
        ),
        L=6,
        rgrid=rgrid,
        tgrid=tgrid,
    )
    test_times = [0.0, 1.0, 5.0]
    test_spatial = [(1.0, 0.5, 0.3), (2.0, 0.1, 1.5)]
    for t in test_times:
        static_at_t = MultipoleExpansionPotential.from_density(
            dens=lambda R, z, phi, _t=t: (
                hp.dens(R, z, use_physical=False)
                * (1.0 + 0.1 * numpy.cos(phi + omega * _t))
            ),
            L=6,
            rgrid=rgrid,
        )
        for R, z, phi in test_spatial:
            # Compare spherical forces directly (both Python)
            fr_s = static_at_t._compute_spher_forces_at_point(R, z, phi)
            fr_t = tdep_mp._compute_spher_forces_at_point(R, z, phi, t=t)
            for i, name in enumerate(["dPhi_dr", "dPhi_dtheta", "dPhi_dphi"]):
                assert (
                    numpy.abs(fr_s[i] - fr_t[i]) < 1e-10 * numpy.abs(fr_s[i]) + 1e-14
                ), (
                    f"{name} mismatch at (R={R}, z={z}, phi={phi}, t={t}): {fr_s[i]} vs {fr_t[i]}"
                )


def test_time_dependent_spline_init():
    """Direct callable (r,t) splines → verify _tdep=True, hasC=True."""
    rgrid = _DEFAULT_RGRID
    hp = HernquistPotential(amp=2.0, a=1.0)
    # Build a static multipole first to get spline shape
    static = MultipoleExpansionPotential.from_density(
        dens=hp, symmetry="spherical", rgrid=rgrid
    )
    # Make callable versions of the splines
    cos_func = lambda r, t: static._rho_cos_splines[0][0](r)
    sin_func = lambda r, t: numpy.zeros_like(r)
    tgrid = numpy.linspace(0, 10, 11)
    tdep = MultipoleExpansionPotential(
        rho_cos_splines=[[cos_func]],
        rho_sin_splines=[[sin_func]],
        rgrid=rgrid,
        tgrid=tgrid,
    )
    assert tdep._tdep is True
    assert tdep.hasC is True
    assert tdep.hasC_dxdv is True
    assert tdep.hasC_dens is True


def test_time_dependent_reduces_to_static():
    """Time-independent density passed via tgrid should match static version."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    rgrid = _DEFAULT_RGRID
    static_mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z: hp.dens(R, z, use_physical=False),
        L=4,
        symmetry="axisymmetric",
        rgrid=rgrid,
    )
    # Same density but with a t parameter that it ignores
    tdep_mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, t=0.0: hp.dens(R, z, use_physical=False),
        L=4,
        symmetry="axisymmetric",
        rgrid=rgrid,
        tgrid=numpy.linspace(0, 10, 11),
    )
    test_points = [(1.0, 0.0), (0.5, 0.3), (2.0, 1.0)]
    for R, z in test_points:
        val_s = static_mp(R, z, use_physical=False)
        val_t = tdep_mp(R, z, t=3.0, use_physical=False)
        assert numpy.abs(val_s - val_t) < 1e-10 * numpy.abs(val_s) + 1e-15, (
            f"Static vs time-dep mismatch at (R={R}, z={z}): {val_s} vs {val_t}"
        )


def test_time_dependent_density_reconstruction():
    """dens() at different times should match input density."""
    from galpy.potential import evaluateDensities

    omega = 1.3
    hp = HernquistPotential(amp=2.0, a=1.0)
    input_dens = lambda R, z, phi, t=0.0: (
        hp.dens(R, z, use_physical=False) * (1.0 + 0.1 * numpy.cos(phi + omega * t))
    )
    tdep_mp = MultipoleExpansionPotential.from_density(
        dens=input_dens,
        L=6,
        rgrid=numpy.geomspace(1e-3, 50, 201),
        tgrid=numpy.linspace(0, 10, 51),
    )
    test_points = [
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.5, 0.3, 2.0),
        (2.0, 0.1, 1.5, 5.0),
    ]
    for R, z, phi, t in test_points:
        dens_input = input_dens(R, z, phi, t)
        dens_recon = evaluateDensities(tdep_mp, R, z, phi=phi, t=t)
        assert (
            numpy.abs(dens_input - dens_recon) < 0.01 * numpy.abs(dens_input) + 1e-10
        ), (
            f"Density mismatch at (R={R}, z={z}, phi={phi}, t={t}): {dens_input} vs {dens_recon}"
        )


def test_time_dependent_isNonAxi_detection():
    """Test axisymmetry detection for time-dependent case."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    omega = 1.3
    # Axisymmetric density with tgrid should detect axisymmetry and truncate
    axi_mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: hp.dens(R, z, use_physical=False),
        L=4,
        rgrid=numpy.geomspace(1e-2, 10, 51),
        tgrid=numpy.linspace(0, 10, 6),
    )
    assert axi_mp.isNonAxi is False
    assert axi_mp._M == 1  # Truncated to axisymmetric
    # Non-axisymmetric density should remain non-axi
    nonaxi_mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False) * (1.0 + 0.1 * numpy.cos(phi + omega * t))
        ),
        L=4,
        rgrid=numpy.geomspace(1e-2, 10, 51),
        tgrid=numpy.linspace(0, 10, 6),
    )
    assert nonaxi_mp.isNonAxi is True
    assert nonaxi_mp._M == 4  # No truncation


def test_time_dependent_hasC():
    """All hasC* flags should be True for time-dependent case (C backend supported)."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    tdep_mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: hp.dens(R, z, use_physical=False),
        L=2,
        rgrid=numpy.geomspace(1e-2, 10, 51),
        tgrid=numpy.linspace(0, 10, 6),
    )
    assert tdep_mp.hasC is True
    assert tdep_mp.hasC_dxdv is True
    assert tdep_mp.hasC_dens is True


def test_time_dependent_tgrid_required():
    """ValueError if time-dependent density but no tgrid."""
    with pytest.raises(ValueError, match="tgrid is required"):
        MultipoleExpansionPotential.from_density(
            dens=lambda R, z, phi, t=0.0: 1.0 / (1.0 + R**2 + z**2),
            L=2,
            rgrid=_DEFAULT_RGRID,
        )


def test_time_dependent_t_keyword_detection():
    """Both f(r, t) and f(r, t=0) should be detected; f(r) should not."""

    hp = HernquistPotential(amp=2.0, a=1.0)
    # f(R, z, phi, t=0) should require tgrid
    with pytest.raises(ValueError, match="tgrid is required"):
        MultipoleExpansionPotential.from_density(
            dens=lambda R, z, phi, t=0.0: hp.dens(R, z, use_physical=False),
            L=2,
            rgrid=_DEFAULT_RGRID,
        )
    # f(R, z) without t should NOT require tgrid (static)
    static_mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z: hp.dens(R, z, use_physical=False),
        L=2,
        symmetry="axisymmetric",
        rgrid=_DEFAULT_RGRID,
    )
    assert static_mp._tdep is False


def test_time_dependent_potential_instance_with_tgrid():
    """from_density with Potential instance and tgrid evaluates at each t."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    # Hernquist is static, so with tgrid it should still be time-dependent
    # but axisymmetric (all m>0 terms zero at all times)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp,
        L=2,
        symmetry="spherical",
        rgrid=_DEFAULT_RGRID,
        tgrid=numpy.linspace(0, 10, 11),
    )
    assert mp._tdep is True
    # Since Hernquist is static, result should match at any t
    static_mp = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID
    )
    for R, z in [(1.0, 0.0), (0.5, 0.3)]:
        val_s = static_mp(R, z, use_physical=False)
        val_t = mp(R, z, t=5.0, use_physical=False)
        assert numpy.abs(val_s - val_t) < 1e-10 * numpy.abs(val_s) + 1e-15, (
            f"Potential+tgrid mismatch at (R={R}, z={z}): {val_s} vs {val_t}"
        )


# --- Time-dependent coverage tests ---

# Shared small grids for fast time-dependent tests
_TDEP_RGRID = numpy.geomspace(1e-2, 10, 21)
_TDEP_TGRID = numpy.linspace(0, 5, 11)


def _make_tdep_axi_mp(L=3, rgrid=_TDEP_RGRID, tgrid=_TDEP_TGRID):
    """Helper: build a small axisymmetric time-dependent MultipoleExpansionPotential."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    return MultipoleExpansionPotential.from_density(
        dens=lambda R, z, t=0.0: hp.dens(R, z, use_physical=False) * (1.0 + 0.1 * t),
        L=L,
        symmetry="axisymmetric",
        rgrid=rgrid,
        tgrid=tgrid,
    )


def _make_tdep_nonaxi_mp(L=3, rgrid=_TDEP_RGRID, tgrid=_TDEP_TGRID):
    """Helper: build a small non-axisymmetric time-dependent MultipoleExpansionPotential."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    return MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False)
            * (1.0 + 0.1 * numpy.cos(phi))
            * (1.0 + 0.05 * t)
        ),
        L=L,
        rgrid=rgrid,
        tgrid=tgrid,
    )


def test_time_dependent_axi_truncation():
    """Time-dep potential with axi density but L=4 should truncate M to 1."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False) * (1.0 + 0.01 * t)
        ),
        L=4,
        rgrid=_TDEP_RGRID,
        tgrid=_TDEP_TGRID,
    )
    assert mp._M == 1, f"Expected M=1 after truncation, got {mp._M}"
    assert mp.isNonAxi is False


def test_init_rejects_callable_cos_splines():
    """Passing a callable as rho_cos_splines should raise ValueError."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    static = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID
    )
    bad_cos = [[lambda r: 0.0]]
    with pytest.raises(ValueError, match="appears to be a callable"):
        MultipoleExpansionPotential(
            rho_cos_splines=bad_cos,
            rgrid=static._rgrid,
        )


def test_init_rejects_callable_sin_splines():
    """Passing a callable as rho_sin_splines should raise ValueError."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    static = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID
    )
    with pytest.raises(ValueError, match="appears to be a callable"):
        MultipoleExpansionPotential(
            rho_cos_splines=static._rho_cos_splines,
            rho_sin_splines=[[lambda r: 0.0]],
            rgrid=static._rgrid,
        )


def test_time_dependent_below_grid():
    """Time-dep evaluation at r < rmin should give finite values."""
    mp = _make_tdep_axi_mp(L=2, rgrid=numpy.geomspace(2.0, 10, 21))
    t_mid = 2.5
    rmin = mp._rgrid[0]
    # R=0.5, z=0 => r=0.5 < rmin=2.0
    val = mp(0.5, 0.0, t=t_mid, use_physical=False)
    assert numpy.isfinite(val), "Potential below grid should be finite"
    rf = mp.Rforce(0.5, 0.0, t=t_mid, use_physical=False)
    assert numpy.isfinite(rf), "Rforce below grid should be finite"
    # Continuity: value at rmin should be close to value just inside
    val_at_rmin = mp(rmin, 0.0, t=t_mid, use_physical=False)
    val_just_inside = mp(rmin * 1.01, 0.0, t=t_mid, use_physical=False)
    assert abs(val_at_rmin - val_just_inside) / abs(val_just_inside) < 0.05


def test_time_dependent_above_grid():
    """Time-dep evaluation at r > rmax should give finite point-mass-like values."""
    mp = _make_tdep_axi_mp(L=2, rgrid=numpy.geomspace(0.01, 5.0, 21))
    rmax = mp._rgrid[-1]
    t = 2.0
    val = mp(8.0, 0.0, t=t, use_physical=False)
    assert numpy.isfinite(val), "Potential above grid should be finite"
    rf = mp.Rforce(8.0, 0.0, t=t, use_physical=False)
    assert numpy.isfinite(rf), "Rforce above grid should be finite"
    # Point-mass decay: potential should decrease in magnitude further out
    val_6 = mp(6.0, 0.0, t=t, use_physical=False)
    val_10 = mp(10.0, 0.0, t=t, use_physical=False)
    assert abs(val_6) > abs(val_10), "Potential should decay with distance"


def test_time_dependent_r_zero():
    """Time-dep evaluation at r=0 should give finite value."""
    mp = _make_tdep_axi_mp(L=2)
    val = mp(0.0, 0.0, t=2.0, use_physical=False)
    assert numpy.isfinite(val), "Potential at r=0 should be finite"


def test_time_dependent_below_grid_l2():
    """Time-dep below-grid with l=2 log branch (L >= 3 needed)."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, t=0.0: (
            hp.dens(R, z, use_physical=False) * (1.0 + 1e-8 * z**2) * (1.0 + 0.1 * t)
        ),
        L=4,
        symmetry="axisymmetric",
        rgrid=numpy.geomspace(2.0, 10, 21),
        tgrid=_TDEP_TGRID,
    )
    # r=sqrt(0.5^2 + 0.5^2) ~ 0.71 < rmin=2.0, L=4 means l=0,1,2,3
    val = mp(0.5, 0.5, t=2.0, use_physical=False)
    rf = mp.Rforce(0.5, 0.5, t=2.0, use_physical=False)
    assert numpy.isfinite(val), "Potential not finite for l=2 below-grid time-dep"
    assert numpy.isfinite(rf), "Rforce not finite for l=2 below-grid time-dep"


def test_time_dependent_2nd_derivs_below_above_grid():
    """Time-dep 2nd derivs at r < rmin and r > rmax (mode=2 path)."""
    mp = _make_tdep_axi_mp(L=2, rgrid=numpy.geomspace(2.0, 5.0, 21))
    t = 2.0
    # Below grid
    r2_below = mp.R2deriv(0.5, 0.0, t=t, use_physical=False)
    z2_below = mp.z2deriv(0.5, 0.0, t=t, use_physical=False)
    assert numpy.isfinite(r2_below), "R2deriv below grid not finite"
    assert numpy.isfinite(z2_below), "z2deriv below grid not finite"
    # Above grid
    r2_above = mp.R2deriv(8.0, 0.0, t=t, use_physical=False)
    z2_above = mp.z2deriv(8.0, 0.0, t=t, use_physical=False)
    assert numpy.isfinite(r2_above), "R2deriv above grid not finite"
    assert numpy.isfinite(z2_above), "z2deriv above grid not finite"


def test_time_dependent_density_at_multiple_times():
    """Density at different times should differ and update _cached_t."""
    from galpy.potential import evaluateDensities

    mp = _make_tdep_axi_mp(L=2)
    t1, t2 = 1.0, 4.0
    d1 = evaluateDensities(mp, 1.0, 0.0, t=t1)
    cached_after_t1 = mp._cached_t
    d2 = evaluateDensities(mp, 1.0, 0.0, t=t2)
    cached_after_t2 = mp._cached_t
    assert cached_after_t1 == t1
    assert cached_after_t2 == t2
    assert d1 != d2, "Density should differ at different times"
    # Density should increase with time since factor is (1 + 0.1*t)
    assert d2 > d1, "Density at t=4 should be larger than at t=1"


def test_time_dependent_array_t_evaluate():
    """Passing array of t values to evaluatePotentials should broadcast correctly."""
    from galpy.potential import evaluatePotentials

    mp = _make_tdep_axi_mp(L=2)
    t_arr = numpy.array([0.0, 1.0, 2.5, 5.0])
    vals = evaluatePotentials(mp, 1.0, 0.0, t=t_arr)
    assert vals.shape == (4,), f"Expected shape (4,), got {vals.shape}"
    assert numpy.all(numpy.isfinite(vals))
    # Values should change with time
    assert not numpy.all(vals == vals[0])


def test_compute_rho_lm_timedep_nonvectorized():
    """Non-vectorizable density should work via fallback loop (non-axi)."""

    class ScalarDensity:
        """Density that doesn't support array broadcasting."""

        def __init__(self):
            self._hp = HernquistPotential(amp=2.0, a=1.0)

        def __call__(self, R, z, phi, t=0.0):
            R = float(R)
            z = float(z)
            phi = float(phi)
            t = float(t)
            return float(
                self._hp.dens(R, z, use_physical=False)
                * (1.0 + 0.1 * numpy.cos(phi))
                * (1.0 + 0.05 * t)
            )

    mp = MultipoleExpansionPotential.from_density(
        dens=ScalarDensity(),
        L=3,
        rgrid=numpy.geomspace(1e-2, 10, 11),
        tgrid=numpy.linspace(0, 5, 5),
    )
    val = mp(1.0, 0.0, phi=0.3, t=2.0, use_physical=False)
    assert numpy.isfinite(val), "Non-vectorized density should produce finite potential"


def test_compute_rho_lm_timedep_axi_nonvectorized():
    """Non-vectorizable axisymmetric density should work via fallback loop."""

    class ScalarDensityAxi:
        """Axisymmetric density that doesn't support array broadcasting."""

        def __init__(self):
            self._hp = HernquistPotential(amp=2.0, a=1.0)

        def __call__(self, R, z, t=0.0):
            R = float(R)
            z = float(z)
            t = float(t)
            return float(self._hp.dens(R, z, use_physical=False) * (1.0 + 0.05 * t))

    mp = MultipoleExpansionPotential.from_density(
        dens=ScalarDensityAxi(),
        L=3,
        symmetry="axisymmetric",
        rgrid=numpy.geomspace(1e-2, 10, 11),
        tgrid=numpy.linspace(0, 5, 5),
    )
    val = mp(1.0, 0.0, t=2.0, use_physical=False)
    assert numpy.isfinite(val), (
        "Non-vectorized axi density should produce finite potential"
    )


def test_time_dependent_force_cache_across_times():
    """Force evaluation at different times should give different results (cache invalidation)."""
    mp = _make_tdep_axi_mp(L=2)
    t1, t2 = 0.5, 4.0
    rf1 = mp.Rforce(1.0, 0.5, t=t1, use_physical=False)
    rf2 = mp.Rforce(1.0, 0.5, t=t2, use_physical=False)
    assert numpy.isfinite(rf1) and numpy.isfinite(rf2)
    assert rf1 != rf2, "Forces at different times should differ"


def test_time_dependent_2nd_derivs():
    """Time-dep 2nd derivatives at a normal grid point."""
    mp = _make_tdep_axi_mp(L=2)
    t = 2.0
    r2 = mp.R2deriv(1.0, 0.5, t=t, use_physical=False)
    z2 = mp.z2deriv(1.0, 0.5, t=t, use_physical=False)
    assert numpy.isfinite(r2), "R2deriv should be finite"
    assert numpy.isfinite(z2), "z2deriv should be finite"


def test_parse_density_time_dependent_2arg():
    """from_density with 2-arg time-dependent density f(R_z, t) should work."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, t=0.0: hp.dens(R, z, use_physical=False) * (1.0 + 0.1 * t),
        L=2,
        symmetry="axisymmetric",
        rgrid=_TDEP_RGRID,
        tgrid=_TDEP_TGRID,
    )
    assert mp._tdep is True
    val = mp(1.0, 0.0, t=2.0, use_physical=False)
    assert numpy.isfinite(val)


def test_time_dependent_forces_r_zero():
    """Time-dep forces at r=0 should return (0, 0, 0)."""
    mp = _make_tdep_axi_mp(L=2)
    dr, dtheta, dphi = mp._compute_spher_forces_at_point(0.0, 0.0, 0.0, t=2.0)
    assert dr == 0.0 and dtheta == 0.0 and dphi == 0.0


# --- Time-dependent non-axisymmetric sin-term coverage tests ---

_OMEGA_SINCOS = 1.3


def _make_tdep_nonaxi_sincos_mp(
    L=3, rgrid=_TDEP_RGRID, tgrid=_TDEP_TGRID, omega=_OMEGA_SINCOS
):
    """Helper: non-axi time-dep potential with both cos and sin harmonics."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    return MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False) * (1.0 + 0.1 * numpy.cos(phi + omega * t))
        ),
        L=L,
        rgrid=rgrid,
        tgrid=tgrid,
    )


def _make_static_snapshot(R, z, phi, t, L=3, rgrid=_TDEP_RGRID, omega=_OMEGA_SINCOS):
    """Build a static MultipoleExpansionPotential matching the time-dep one frozen at time t."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    return MultipoleExpansionPotential.from_density(
        dens=lambda R_, z_, phi_: (
            hp.dens(R_, z_, use_physical=False)
            * (1.0 + 0.1 * numpy.cos(phi_ + omega * t))
        ),
        L=L,
        rgrid=rgrid,
    )


def test_time_dependent_nonaxi_potential_correctness():
    """Non-axi time-dep potential with sin harmonics matches static snapshot."""
    mp = _make_tdep_nonaxi_sincos_mp(L=3)
    pts = [(1.0, 0.5, 0.3), (2.0, 0.1, 1.5)]
    for t in [0.0, 2.5]:
        static = _make_static_snapshot(0, 0, 0, t, L=3)
        for R, z, phi in pts:
            val_td = mp(R, z, phi=phi, t=t, use_physical=False)
            val_st = static(R, z, phi=phi, use_physical=False)
            assert abs(val_td - val_st) / abs(val_st) < 0.02, (
                f"Potential mismatch at R={R}, z={z}, phi={phi}, t={t}: "
                f"td={val_td}, st={val_st}"
            )


def test_time_dependent_nonaxi_forces_correctness():
    """Non-axi time-dep forces with sin harmonics match static snapshot."""
    mp = _make_tdep_nonaxi_sincos_mp(L=3)
    t = 2.5
    static = _make_static_snapshot(0, 0, 0, t, L=3)
    R, z, phi = 1.0, 0.5, 0.7
    r = numpy.sqrt(R**2 + z**2)
    costheta = z / r
    sintheta = R / r
    td_forces = mp._compute_spher_forces_at_point(r, costheta, phi, t=t)
    st_forces = static._compute_spher_forces_at_point(r, costheta, phi)
    for i, name in enumerate(["dr", "dtheta", "dphi"]):
        scale = max(abs(st_forces[i]), 1e-10)
        assert abs(td_forces[i] - st_forces[i]) / scale < 0.02, (
            f"Force {name} mismatch at t={t}: td={td_forces[i]}, st={st_forces[i]}"
        )


def test_time_dependent_nonaxi_2nd_derivs():
    """Non-axi time-dep 2nd derivatives with sin harmonics match static snapshot."""
    mp = _make_tdep_nonaxi_sincos_mp(L=3)
    t = 2.5
    static = _make_static_snapshot(0, 0, 0, t, L=3)
    R, z, phi = 1.0, 0.5, 0.7
    for deriv_name in ["R2deriv", "z2deriv", "Rzderiv", "phi2deriv", "Rphideriv"]:
        method_td = getattr(mp, deriv_name)
        method_st = getattr(static, deriv_name)
        val_td = method_td(R, z, phi=phi, t=t, use_physical=False)
        val_st = method_st(R, z, phi=phi, use_physical=False)
        scale = max(abs(val_st), 1e-10)
        assert abs(val_td - val_st) / scale < 0.05, (
            f"{deriv_name} mismatch at t={t}: td={val_td}, st={val_st}"
        )


def test_time_dependent_nonaxi_below_grid():
    """Non-axi time-dep eval below grid with sin interpolators."""
    mp = _make_tdep_nonaxi_sincos_mp(L=3, rgrid=numpy.geomspace(2.0, 10, 21))
    t = 2.5
    R, z, phi = 0.5, 0.3, 0.7
    # r ~ 0.58 < rmin=2.0
    val = mp(R, z, phi=phi, t=t, use_physical=False)
    assert numpy.isfinite(val), "Potential below grid not finite"
    rf = mp.Rforce(R, z, phi=phi, t=t, use_physical=False)
    assert numpy.isfinite(rf), "Rforce below grid not finite"
    r2 = mp.R2deriv(R, z, phi=phi, t=t, use_physical=False)
    z2 = mp.z2deriv(R, z, phi=phi, t=t, use_physical=False)
    assert numpy.isfinite(r2), "R2deriv below grid not finite"
    assert numpy.isfinite(z2), "z2deriv below grid not finite"


def test_time_dependent_nonaxi_above_grid():
    """Non-axi time-dep eval above grid with sin interpolators."""
    mp = _make_tdep_nonaxi_sincos_mp(L=3, rgrid=numpy.geomspace(0.01, 5.0, 21))
    t = 2.5
    phi = 0.7
    val_6 = mp(6.0, 0.0, phi=phi, t=t, use_physical=False)
    val_10 = mp(10.0, 0.0, phi=phi, t=t, use_physical=False)
    assert numpy.isfinite(val_6), "Potential above grid not finite"
    assert numpy.isfinite(val_10), "Potential above grid not finite"
    assert abs(val_6) > abs(val_10), "Potential should decay with distance"
    rf = mp.Rforce(8.0, 0.0, phi=phi, t=t, use_physical=False)
    assert numpy.isfinite(rf), "Rforce above grid not finite"
    r2 = mp.R2deriv(8.0, 0.0, phi=phi, t=t, use_physical=False)
    assert numpy.isfinite(r2), "R2deriv above grid not finite"


def test_time_dependent_nonaxi_pole_clamping():
    """Non-axi time-dep 2nd derivs at pole (R=0) should be finite (pole clamping)."""
    mp = _make_tdep_nonaxi_sincos_mp(L=3)
    t = 2.5
    for z in [1.0, -1.0]:
        r2 = mp.R2deriv(0.0, z, phi=0.5, t=t, use_physical=False)
        z2 = mp.z2deriv(0.0, z, phi=0.5, t=t, use_physical=False)
        rz = mp.Rzderiv(0.0, z, phi=0.5, t=t, use_physical=False)
        assert numpy.isfinite(r2), f"R2deriv at pole z={z} not finite"
        assert numpy.isfinite(z2), f"z2deriv at pole z={z} not finite"
        assert numpy.isfinite(rz), f"Rzderiv at pole z={z} not finite"


def test_time_dependent_nonaxi_density_reconstruction():
    """Non-axi time-dep density with sin harmonics reconstructs input density."""
    from galpy.potential import evaluateDensities

    hp = HernquistPotential(amp=2.0, a=1.0)
    omega = _OMEGA_SINCOS
    rgrid = numpy.geomspace(1e-2, 10, 41)
    tgrid = numpy.linspace(0, 5, 21)
    mp = _make_tdep_nonaxi_sincos_mp(L=5, rgrid=rgrid, tgrid=tgrid)
    t = 2.5
    pts = [(1.0, 0.0, 0.7), (2.0, 0.5, 1.2), (0.5, 0.3, 0.3)]
    for R, z, phi in pts:
        d_mp = evaluateDensities(mp, R, z, phi=phi, t=t, use_physical=False)
        d_ref = hp.dens(R, z, use_physical=False) * (
            1.0 + 0.1 * numpy.cos(phi + omega * t)
        )
        assert abs(d_mp - d_ref) / abs(d_ref) < 0.05, (
            f"Density mismatch at R={R}, z={z}, phi={phi}, t={t}: "
            f"mp={d_mp}, ref={d_ref}"
        )


def test_parse_density_1arg_time_dependent():
    """_parse_density with 1-arg time-dep density f(r, t) should set _tdep=True."""
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda r, t=0.0: 1.0 / (1.0 + r) ** 4 * (1.0 + 0.1 * t),
        L=2,
        symmetry="spherical",
        rgrid=_TDEP_RGRID,
        tgrid=_TDEP_TGRID,
    )
    assert mp._tdep is True
    v1 = mp(1.0, 0.0, t=0.0, use_physical=False)
    v2 = mp(1.0, 0.0, t=3.0, use_physical=False)
    assert numpy.isfinite(v1) and numpy.isfinite(v2)
    assert v1 != v2, "Potential should change with time"


def test_time_dependent_nonaxi_c_orbit():
    """Orbit integration with C for non-axi time-dep potential with sin harmonics."""
    from galpy.orbit import Orbit

    mp = _make_tdep_nonaxi_sincos_mp(L=3)
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.05, 0.3])
    ts = numpy.linspace(0, 2, 51)
    o.integrate(ts, mp, method="dop853_c")
    assert numpy.all(numpy.isfinite(o.R(ts))), "Orbit R not finite"
    assert numpy.all(numpy.isfinite(o.z(ts))), "Orbit z not finite"
    assert numpy.all(numpy.isfinite(o.phi(ts))), "Orbit phi not finite"


def test_time_dependent_nonaxi_forces_r_zero():
    """Non-axi time-dep forces at r=0 should return (0, 0, 0)."""
    mp = _make_tdep_nonaxi_sincos_mp(L=3)
    dr, dtheta, dphi = mp._compute_spher_forces_at_point(0.0, 0.0, 0.0, t=2.0)
    assert dr == 0.0 and dtheta == 0.0 and dphi == 0.0


def test_time_dependent_nonaxi_array_t():
    """Non-axi time-dep potential with array of t and phi!=0 should broadcast."""
    from galpy.potential import evaluatePotentials

    mp = _make_tdep_nonaxi_sincos_mp(L=3)
    t_arr = numpy.array([0.0, 1.0, 2.5, 5.0])
    vals = evaluatePotentials(mp, 1.0, 0.5, phi=0.7, t=t_arr)
    assert vals.shape == (4,), f"Expected shape (4,), got {vals.shape}"
    assert numpy.all(numpy.isfinite(vals))
    assert not numpy.all(vals == vals[0]), "Values should vary with time"


# --- Additional coverage tests ---


def test_tdep_init_sin_splines_none():
    """Time-dep __init__ with rho_sin_splines=None should create zero sin funcs."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    temp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False)
            * (1.0 + 0.1 * numpy.cos(phi))
            * (1.0 + 0.05 * t)
        ),
        L=3,
        rgrid=_TDEP_RGRID,
        tgrid=_TDEP_TGRID,
    )
    # Construct via __init__ with rho_sin_splines=None
    mp = MultipoleExpansionPotential(
        rho_cos_splines=temp._rho_cos_funcs,
        rho_sin_splines=None,
        rgrid=temp._rgrid,
        tgrid=temp._tgrid,
    )
    # Should still work and produce finite values
    val = mp(1.0, 0.5, phi=0.3, t=1.0, use_physical=False)
    assert numpy.isfinite(val), "Potential should be finite with sin_splines=None"


def test_tdep_sin_only_harmonics_detection():
    """Time-dep potential with sin(phi) density detects non-axi via sin harmonics."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False)
            * (1.0 + 0.1 * numpy.sin(phi))
            * (1.0 + 0.05 * t)
        ),
        L=3,
        rgrid=_TDEP_RGRID,
        tgrid=_TDEP_TGRID,
    )
    assert mp.isNonAxi is True, "sin(phi) density should be detected as non-axi"
    # Verify potential differs at different phi
    v1 = mp(1.0, 0.0, phi=0.0, t=1.0, use_physical=False)
    v2 = mp(1.0, 0.0, phi=1.0, t=1.0, use_physical=False)
    assert v1 != v2, "Potential should vary with phi"


def test_init_rejects_non_spline_sin_entry():
    """Passing a non-callable, non-spline sin entry should raise TypeError."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    static = MultipoleExpansionPotential.from_density(
        dens=hp, L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID
    )
    with pytest.raises(TypeError, match="must be an InterpolatedUnivariateSpline"):
        MultipoleExpansionPotential(
            rho_cos_splines=static._rho_cos_splines,
            rho_sin_splines=[[42]],
            rgrid=static._rgrid,
        )


def test_time_dependent_nonaxi_c_orbit_limited_grid():
    """C orbit integration with limited grid triggers below/above grid C paths."""
    from galpy.orbit import Orbit

    mp = _make_tdep_nonaxi_sincos_mp(L=3, rgrid=numpy.geomspace(1.1, 2.0, 21))
    # Start orbit at R=1.0 with enough velocity to leave the grid
    o = Orbit([1.0, 0.3, 1.1, 0.0, 0.2, 0.0])
    ts = numpy.linspace(0, 2, 51)
    o.integrate(ts, mp, method="dop853_c")
    assert numpy.all(numpy.isfinite(o.R(ts))), "Orbit R not finite"
    assert numpy.all(numpy.isfinite(o.z(ts))), "Orbit z not finite"
    # Also test dxdv (2nd derivs) with below-grid: planar orbit
    o2 = Orbit([1.0, 0.3, 1.1, 0.3])
    o2.integrate_dxdv([1e-8, 0.0, 0.0, 0.0], ts, mp, method="dop853_c")
    assert numpy.all(numpy.isfinite(o2.getOrbit_dxdv())), "dxdv below grid not finite"


def test_time_dependent_nonaxi_c_dxdv():
    """dxdv integration exercises C 2nd derivative code paths."""
    from galpy.orbit import Orbit

    mp = _make_tdep_nonaxi_sincos_mp(L=3)
    # integrate_dxdv requires 4D (planar) orbit
    o = Orbit([1.0, 0.1, 1.1, 0.3])
    ts = numpy.linspace(0, 1, 21)
    o.integrate_dxdv([1e-8, 0.0, 0.0, 0.0], ts, mp, method="dop853_c")
    dxdv = o.getOrbit_dxdv()
    assert numpy.all(numpy.isfinite(dxdv)), "dxdv should be finite"


def test_time_dependent_nonaxi_phizderiv():
    """phizderiv on non-axi time-dep potential should be finite."""
    mp = _make_tdep_nonaxi_sincos_mp(L=3)
    val = mp.phizderiv(1.0, 0.5, phi=0.7, t=2.5, use_physical=False)
    assert numpy.isfinite(val), "phizderiv should be finite"


def test_time_dependent_nonaxi_array_density():
    """Array density evaluation (broadcast path) for non-axi time-dep potential."""
    from galpy.potential import evaluateDensities

    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = _make_tdep_nonaxi_sincos_mp(L=3)
    R_arr = numpy.array([0.5, 1.0, 2.0])
    d = evaluateDensities(mp, R_arr, 0.0, phi=0.5, t=1.0, use_physical=False)
    assert d.shape == (3,), f"Expected shape (3,), got {d.shape}"
    assert numpy.all(numpy.isfinite(d))


def test_static_nonaxi_nonvectorized_density():
    """Static non-axi potential from non-vectorizable density (fallback loop)."""

    class ScalarDens:
        """Density that doesn't support array broadcasting."""

        def __init__(self):
            self._hp = HernquistPotential(amp=2.0, a=1.0)

        def __call__(self, R, z, phi):
            R = float(R)
            z = float(z)
            phi = float(phi)
            return float(
                self._hp.dens(R, z, use_physical=False) * (1.0 + 0.1 * numpy.cos(phi))
            )

    mp = MultipoleExpansionPotential.from_density(
        dens=ScalarDens(),
        L=3,
        rgrid=numpy.geomspace(1e-2, 10, 11),
    )
    val = mp(1.0, 0.0, phi=0.3, use_physical=False)
    assert numpy.isfinite(val), (
        "Non-vectorized static density should give finite potential"
    )


def test_time_dependent_c_density_dynamical_friction():
    """ChandrasekharDynamicalFrictionForce with time-dep multipole density exercises C densityc."""
    from galpy.orbit import Orbit
    from galpy.potential import (
        ChandrasekharDynamicalFrictionForce,
        LogarithmicHaloPotential,
    )

    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False) * (1 + 1e-6 * t)
        ),
        L=2,
        symmetry="spherical",
        rgrid=numpy.geomspace(1e-2, 20, 51),
        tgrid=numpy.linspace(0, 10, 11),
    )
    lp = LogarithmicHaloPotential(normalize=1.0)
    cdf = ChandrasekharDynamicalFrictionForce(
        GMs=0.01,
        dens=mp,
        sigmar=lambda r: 1.0 / numpy.sqrt(2.0),
    )
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.05, 0.3])
    ts = numpy.linspace(0.0, 1.0, 51)
    o.integrate(ts, lp + cdf, method="dop853_c")
    assert numpy.all(numpy.isfinite(o.R(ts))), "Orbit R not finite"
    assert numpy.all(numpy.isfinite(o.z(ts))), "Orbit z not finite"


def test_finalize_pot_args_trailing_scalars():
    """Planar orbit with time-dep MEP + simple potential covers trailing-scalar flush in _finalize_pot_args."""
    from galpy.orbit import Orbit
    from galpy.potential import PlummerPotential

    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False) * (1 + 1e-6 * t)
        ),
        L=2,
        symmetry="spherical",
        rgrid=numpy.geomspace(1e-2, 20, 51),
        tgrid=numpy.linspace(0, 10, 11),
    )
    pp = PlummerPotential(amp=0.1, b=0.5)
    o = Orbit([1.0, 0.1, 1.1, 0.3])
    ts = numpy.linspace(0, 1, 21)
    o.integrate(ts, mp + pp, method="dop853_c")
    assert numpy.all(numpy.isfinite(o.R(ts))), "Orbit R not finite"


def _make_timedep_disk_mep(amp=1.0):
    """Helper: DiskMEP with time-dep inner ._me for coverage of ndarray serialization branches."""
    from galpy.potential import DiskMultipoleExpansionPotential

    hp = HernquistPotential(amp=2.0, a=1.0)
    dmep = DiskMultipoleExpansionPotential(
        amp=amp,
        dens=lambda R, z: (
            13.5 * numpy.exp(-3.0 * R) * numpy.exp(-27.0 * numpy.fabs(z))
            + hp.dens(R, z, use_physical=False)
        ),
        Sigma={"h": 1.0 / 3.0, "type": "exp", "amp": 1.0},
        hz={"type": "exp", "h": 1.0 / 27.0},
        L=3,
        rgrid=numpy.geomspace(1e-2, 20, 51),
    )
    # Replace the static inner MultipoleExpansionPotential with a time-dep one
    # so that _serialize_for_c() returns ndarray instead of list
    static_me = dmep._me
    tdep_me = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            static_me.dens(R, z, use_physical=False) * (1 + 1e-8 * t)
        ),
        L=static_me._L,
        rgrid=numpy.geomspace(1e-2, 20, 51),
        tgrid=numpy.linspace(0, 10, 11),
        symmetry="spherical",
    )
    dmep._me = tdep_me
    return dmep


def test_diskmep_timedep_parse_pot_full():
    """_parse_pot for full orbit with time-dep DiskMEP exercises ndarray append branch."""
    from galpy.orbit.integrateFullOrbit import _parse_pot

    dmep = _make_timedep_disk_mep()
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot([dmep])
    assert npot >= 2
    assert pot_args.dtype == numpy.float64


def test_diskmep_timedep_parse_pot_planar():
    """_parse_pot for planar orbit with time-dep DiskMEP exercises ndarray append branch."""
    from galpy.orbit.integratePlanarOrbit import _parse_pot
    from galpy.potential.planarPotential import toPlanarPotential

    dmep = _make_timedep_disk_mep()
    planar = toPlanarPotential(dmep)
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot([planar])
    assert npot >= 2
    assert pot_args.dtype == numpy.float64


def test_diskmep_timedep_parse_pot_linear():
    """_parse_pot for linear orbit with time-dep DiskMEP exercises ndarray append branch."""
    from galpy.orbit.integrateLinearOrbit import _parse_pot
    from galpy.potential.verticalPotential import toVerticalPotential

    dmep = _make_timedep_disk_mep()
    vert = toVerticalPotential(dmep, 1.0)
    npot, pot_type, pot_args, pot_tfuncs = _parse_pot([vert])
    assert npot >= 2
    assert pot_args.dtype == numpy.float64


def test_diskmep_timedep_extra_amp_parse_pot():
    """_parse_multipole_expansion_pot with time-dep MEP + extra_amp exercises ndarray copy branch."""
    from galpy.orbit.integratePlanarOrbit import _parse_multipole_expansion_pot

    hp = HernquistPotential(amp=2.0, a=1.0)
    tdep_me = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False) * (1 + 1e-8 * t)
        ),
        L=2,
        symmetry="spherical",
        rgrid=numpy.geomspace(1e-2, 20, 51),
        tgrid=numpy.linspace(0, 10, 11),
    )
    original = tdep_me._serialize_for_c().copy()
    pt, pa = _parse_multipole_expansion_pot(tdep_me, extra_amp=2.0)
    assert pt == 44
    assert isinstance(pa, numpy.ndarray)
    # extra_amp should have been applied (amp at index 4+Nr doubled)
    Nr = int(pa[0])
    assert pa[4 + Nr] == pytest.approx(original[4 + Nr] * 2.0)
    # Original should not be mutated
    assert numpy.array_equal(tdep_me._serialize_for_c(), original)


def test_time_dependent_quantity_density_warning():
    """Time-dep density returning astropy Quantity should warn about unsupported units."""
    pytest.importorskip("astropy")
    import warnings

    from astropy import units

    from galpy.util import galpyWarning

    hp = HernquistPotential(amp=2.0, a=1.0)
    dens_with_units = lambda R, z, phi, t=0.0: (
        hp.dens(R, z, use_physical=False) * (1 + 1e-6 * t) * units.Msun / units.pc**3
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            MultipoleExpansionPotential.from_density(
                dens=dens_with_units,
                L=2,
                symmetry="spherical",
                rgrid=numpy.geomspace(1e-2, 20, 51),
                tgrid=numpy.linspace(0, 10, 5),
            )
        except Exception:
            pass  # Expected: Quantity density causes downstream errors
    galpy_warnings = [x for x in w if issubclass(x.category, galpyWarning)]
    assert len(galpy_warnings) >= 1, "Expected warning about Quantity density"
    assert "time-dependent" in str(galpy_warnings[0].message).lower()


def test_large_L_c_orbit():
    """Orbit integration with L > 64 verifies no stack buffer overflow from old MEP_MAX_LM limit."""
    from galpy.orbit import Orbit
    from galpy.potential import evaluatePotentials, evaluateRforces, evaluatezforces

    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential.from_density(
        dens=hp,
        L=80,
        symmetry="spherical",
        rgrid=numpy.geomspace(1e-2, 20, 101),
    )
    # Evaluate potential and forces at a test point
    val = evaluatePotentials(mp, 1.0, 0.5, use_physical=False)
    rf = evaluateRforces(mp, 1.0, 0.5, use_physical=False)
    zf = evaluatezforces(mp, 1.0, 0.5, use_physical=False)
    assert numpy.isfinite(val)
    assert numpy.isfinite(rf)
    assert numpy.isfinite(zf)
    # Integrate orbit
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.05, 0.3])
    ts = numpy.linspace(0, 1, 21)
    o.integrate(ts, mp, method="dop853_c")
    assert numpy.all(numpy.isfinite(o.R(ts))), "Orbit R not finite"
    assert numpy.all(numpy.isfinite(o.z(ts))), "Orbit z not finite"
