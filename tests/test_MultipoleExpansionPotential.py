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
    mp = MultipoleExpansionPotential(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    for R in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        assert abs(mp(R, 0.0) - hp(R, 0.0)) / abs(hp(R, 0.0)) < 0.01, (
            f"Potential mismatch at R={R}"
        )


def test_spherical_potential_off_plane():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    pts = [(1.0, 0.5), (0.5, 1.0), (2.0, 1.0)]
    for R, z in pts:
        assert abs(mp(R, z) - hp(R, z)) / abs(hp(R, z)) < 0.01, (
            f"Potential mismatch at R={R}, z={z}"
        )


def test_spherical_density_matches_hernquist():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    for R in [0.1, 0.5, 1.0, 2.0, 5.0]:
        d_hp = hp.dens(R, 0.0)
        d_mp = mp.dens(R, 0.0)
        assert abs(d_mp - d_hp) / abs(d_hp) < 0.01, (
            f"Density mismatch at R={R}: hp={d_hp}, mp={d_mp}"
        )


def test_spherical_isNonAxi_false():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    assert not mp.isNonAxi


# --- Axisymmetric tests (MiyamotoNagai) ---


def test_axisymmetric_potential_matches_mn():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential(
        dens=mn, L=16, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    pts = [(1.0, 0.0), (1.0, 0.5), (2.0, 0.1), (0.5, 0.5)]
    for R, z in pts:
        assert abs(mp(R, z) - mn(R, z)) / abs(mn(R, z)) < 0.02, (
            f"Potential mismatch at R={R}, z={z}"
        )


def test_axisymmetric_density_matches_mn_midplane():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential(
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
    mp = MultipoleExpansionPotential(
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
    mp = MultipoleExpansionPotential(
        dens=scf, L=6, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    pts = [(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (2.0, 1.0)]
    for R, z in pts:
        v_scf = scf(R, z)
        v_mp = mp(R, z)
        assert abs(v_mp - v_scf) / abs(v_scf) < 0.02, (
            f"Potential mismatch at R={R}, z={z}: scf={v_scf}, mp={v_mp}"
        )


def test_scf_density_cross_validation():
    Acos = numpy.zeros((3, 3, 1))
    Acos[0, 0, 0] = 1.0
    Acos[1, 0, 0] = 0.1
    Acos[0, 1, 0] = 0.05
    scf = SCFPotential(Acos=Acos, a=1.0)
    mp = MultipoleExpansionPotential(
        dens=scf, L=6, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    pts = [(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (2.0, 0.0)]
    for R, z in pts:
        d_scf = scf.dens(R, z)
        d_mp = mp.dens(R, z)
        if abs(d_scf) > 1e-10:
            assert abs(d_mp - d_scf) / abs(d_scf) < 0.02, (
                f"Density mismatch at R={R}, z={z}: scf={d_scf}, mp={d_mp}"
            )


# --- Density reconstruction ---


def test_spherical_density_reconstruction():
    coeff = 1.0 / (2.0 * numpy.pi)

    def dens(r):
        return coeff / r / (1 + r) ** 3

    mp = MultipoleExpansionPotential(
        dens=dens, L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID
    )
    for R in [0.1, 0.5, 1.0, 2.0, 5.0]:
        d_true = dens(R)
        d_mp = mp.dens(R, 0.0)
        assert abs(d_mp - d_true) / abs(d_true) < 0.01, (
            f"Density reconstruction failed at R={R}: true={d_true}, mp={d_mp}"
        )


def test_axisymmetric_density_reconstruction():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential(
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
    mp = MultipoleExpansionPotential(
        L=2, symmetry="spherical", normalize=True, rgrid=_FINE_RGRID
    )
    vc = mp.vcirc(1.0, 0.0)
    assert abs(vc - 1.0) < 0.02, f"vcirc(1,0) = {vc}, expected ~1.0"


def test_normalize_fraction():
    mp = MultipoleExpansionPotential(
        L=2, symmetry="spherical", normalize=0.5, rgrid=_FINE_RGRID
    )
    vc = mp.vcirc(1.0, 0.0)
    assert abs(vc - numpy.sqrt(0.5)) < 0.02, (
        f"vcirc(1,0) = {vc}, expected ~{numpy.sqrt(0.5)}"
    )


# --- isNonAxi ---


def test_spherical_is_axi():
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical")
    assert not mp.isNonAxi


def test_axisymmetric_is_axi():
    mp = MultipoleExpansionPotential(L=6, symmetry="axisymmetric")
    assert not mp.isNonAxi


def test_general_with_axi_density_is_axi():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential(dens=hp, L=4, symmetry=None)
    assert not mp.isNonAxi


# --- Density input variants ---


def test_potential_instance_input():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.01


def test_2arg_lambda_input():
    # rho = amp/(4*pi) * a / (r * (r+a)^3) for HernquistPotential(amp=2, a=1)
    coeff = 1.0 / (2.0 * numpy.pi)
    mp = MultipoleExpansionPotential(
        dens=lambda R, z: coeff
        / numpy.sqrt(R**2 + z**2)
        / (1 + numpy.sqrt(R**2 + z**2)) ** 3,
        L=2,
        symmetry="spherical",
        rgrid=_FINE_RGRID,
    )
    hp = HernquistPotential(amp=2.0, a=1.0)
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.01


def test_1arg_lambda_input():
    coeff = 1.0 / (2.0 * numpy.pi)
    mp = MultipoleExpansionPotential(
        dens=lambda r: coeff / r / (1 + r) ** 3,
        L=2,
        symmetry="spherical",
        rgrid=_FINE_RGRID,
    )
    hp = HernquistPotential(amp=2.0, a=1.0)
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.01


# --- Edge cases ---


def test_r_zero():
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical", rgrid=_FINE_RGRID)
    val = mp(0.0, 0.0)
    assert numpy.isfinite(val)


def test_monopole_only():
    mp = MultipoleExpansionPotential(L=1, symmetry="spherical", rgrid=_FINE_RGRID)
    hp = HernquistPotential(amp=2.0, a=1.0)
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.01


def test_OmegaP_zero():
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical")
    assert mp.OmegaP() == 0


def test_hasC():
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical")
    assert mp.hasC
    assert mp.hasC_dxdv
    assert mp.hasC_dens


def test_default_rgrid():
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical")
    val = mp(1.0, 0.0)
    assert numpy.isfinite(val)


# --- Analytical force tests ---


def test_spherical_Rforce():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    for R in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        rf_mp = mp.Rforce(R, 0.0)
        rf_hp = hp.Rforce(R, 0.0)
        assert abs(rf_mp - rf_hp) / abs(rf_hp) < 0.02, (
            f"Rforce mismatch at R={R}: mp={rf_mp}, hp={rf_hp}"
        )


def test_spherical_zforce():
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    pts = [(1.0, 0.5), (0.5, 1.0), (2.0, 1.0)]
    for R, z in pts:
        zf_mp = mp.zforce(R, z)
        zf_hp = hp.zforce(R, z)
        assert abs(zf_mp - zf_hp) / abs(zf_hp) < 0.02, (
            f"zforce mismatch at R={R}, z={z}: mp={zf_mp}, hp={zf_hp}"
        )


def test_axisymmetric_Rforce():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential(
        dens=mn, L=16, symmetry="axisymmetric", rgrid=_FINE_RGRID
    )
    pts = [(1.0, 0.0), (1.0, 0.5), (2.0, 0.1), (0.5, 0.5)]
    for R, z in pts:
        rf_mp = mp.Rforce(R, z)
        rf_mn = mn.Rforce(R, z)
        assert abs(rf_mp - rf_mn) / abs(rf_mn) < 0.02, (
            f"Rforce mismatch at R={R}, z={z}: mp={rf_mp}, mn={rf_mn}"
        )


def test_axisymmetric_zforce():
    mn = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.5)
    mp = MultipoleExpansionPotential(
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
    mp = MultipoleExpansionPotential(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID
    )
    for R in [0.5, 1.0, 2.0]:
        pt = mp.phitorque(R, 0.0, phi=0.5)
        assert abs(pt) < 1e-10, f"phitorque not zero at R={R}: {pt}"


# --- Coverage: density input variants ---


def test_3arg_callable_density_input():
    """Test that a 3-argument callable density (R, z, phi) without units works."""
    coeff = 1.0 / (2.0 * numpy.pi)
    mp = MultipoleExpansionPotential(
        dens=lambda R, z, phi: coeff
        / numpy.sqrt(R**2 + z**2)
        / (1 + numpy.sqrt(R**2 + z**2)) ** 3,
        L=4,
        symmetry=None,
        rgrid=_DEFAULT_RGRID,
    )
    hp = HernquistPotential(amp=2.0, a=1.0)
    assert abs(mp(1.0, 0.0) - hp(1.0, 0.0)) / abs(hp(1.0, 0.0)) < 0.02


def test_dens_phi_none():
    """Test that _dens handles phi=None for axisymmetric potential."""
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID)
    val = mp._dens(1.0, 0.0, phi=None)
    assert numpy.isfinite(val) and val > 0


def test_dens_at_infinity():
    """Test that density at r=infinity returns 0."""
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID)
    val = mp.dens(numpy.inf, 0.0, use_physical=False)
    assert val == 0.0


def test_spher_forces_at_r_zero():
    """Test that spherical force components at r=0 return 0."""
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID)
    dr, dtheta, dphi = mp._compute_spher_forces_at_point(0.0, 0.0, 0.0)
    assert dr == 0.0 and dtheta == 0.0 and dphi == 0.0


def test_spher_forces_at_infinity():
    """Test that spherical force components at r=infinity return 0."""
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID)
    dr, dtheta, dphi = mp._compute_spher_forces_at_point(numpy.inf, 0.0, 0.0)
    assert dr == 0.0 and dtheta == 0.0 and dphi == 0.0


# --- Second derivative tests ---


def test_2nd_derivs_at_r_zero():
    """Test that second derivatives at r=0 return all zeros."""
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID)
    result = mp._compute_spher_2nd_derivs_at_point(0.0, 0.0, 0.0)
    assert all(v == 0.0 for v in result)
    # Also test through the public interface (hits _cyl_2nd_deriv_at_point r=0 path)
    assert mp.R2deriv(0.0, 0.0, use_physical=False) == 0.0
    assert mp.z2deriv(0.0, 0.0, use_physical=False) == 0.0


def test_2nd_derivs_at_infinity():
    """Test that second derivatives at r=infinity return all zeros."""
    mp = MultipoleExpansionPotential(L=2, symmetry="spherical", rgrid=_DEFAULT_RGRID)
    result = mp._compute_spher_2nd_derivs_at_point(numpy.inf, 0.0, 0.0)
    assert all(v == 0.0 for v in result)


def test_2nd_derivs_on_z_axis():
    """Test that second derivatives are finite on the z-axis (R=0, costheta=±1)
    where dP/d(costheta) diverges for m>0, triggering the pole clamping."""
    coeff = 1.0 / (2.0 * numpy.pi)
    mp = MultipoleExpansionPotential(
        dens=lambda R, z, phi: coeff
        / numpy.sqrt(R**2 + z**2)
        / (1 + numpy.sqrt(R**2 + z**2)) ** 3
        * (1.0 + 0.1 * numpy.cos(2 * phi)),
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
    mp = MultipoleExpansionPotential(dens=hp, L=6, symmetry=None, rgrid=_FINE_RGRID)
    z = 1.0
    R2_axis = mp.R2deriv(0.0, z, use_physical=False)
    R2_near = mp.R2deriv(1e-4, z, use_physical=False)
    assert abs(R2_axis - R2_near) / abs(R2_near) < 0.01, (
        f"R2deriv discontinuous at z-axis: on_axis={R2_axis}, near={R2_near}"
    )


def test_spherical_2nd_derivs_match_hernquist():
    """Test that second derivatives match Hernquist for a spherical expansion."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential(
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
            assert abs(val_mp - val_hp) / abs(val_hp) < 0.02, (
                f"{name} mismatch at R={R}, z={z}: mp={val_mp}, hp={val_hp}"
            )


def test_spline_degree_k_parameter():
    """Test that the k parameter is passed through to splines."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp3 = MultipoleExpansionPotential(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID, k=3
    )
    mp5 = MultipoleExpansionPotential(
        dens=hp, L=2, symmetry="spherical", rgrid=_FINE_RGRID, k=5
    )
    assert mp3._k == 3
    assert mp5._k == 5
    # Both should give reasonable results
    for mp in [mp3, mp5]:
        val = mp.R2deriv(1.0, 0.5, use_physical=False)
        assert numpy.isfinite(val)


# --- Below/above grid extrapolation tests ---


def test_below_grid_potential_force_2ndderiv():
    """Test that potential, forces, and second derivatives are finite and
    well-behaved below the grid (r < rmin), covering the constant-density
    extrapolation in _eval_R_lm, _eval_dR_lm, _eval_d2R_lm."""
    hp = HernquistPotential(amp=2.0, a=1.0)
    mp = MultipoleExpansionPotential(
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
    mp = MultipoleExpansionPotential(
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
    mp = MultipoleExpansionPotential(
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
    mp = MultipoleExpansionPotential(
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
