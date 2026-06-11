def _liouville3d_tdep_amp(t):
    # Smooth, strictly-positive time-dependent amplitude used by the
    # TimeDependentAmplitudeWrapperPotential registry entry (module-level so it
    # is shared identically by the C and pure-Python integration paths).
    import numpy

    return 1.0 + 0.3 * numpy.sin(t)


def pytest_generate_tests(metafunc):
    # galpy imports must be hear to not interfere with different config settings
    # in different files
    # Maybe I should define a cmdline option to set the config instead...
    import numpy

    from galpy import potential

    if metafunc.function.__name__ in (
        "test_liouville_3d",
        "test_liouville_3d_2d_bridge",
        "test_dxdv_3d_c_vs_python",
    ):
        # Single CATEGORIZED registry of EVERY potential that currently advertises a
        # complete 3D C Hessian (hasC_dxdv3d=True). Each entry is
        # (potential_instance, id_string, category) with
        # category in {"spherical", "axisymmetric", "nonaxisymmetric"}. The
        # parametrized 3D variational tests (det(M)/symplecticity/flow/FD-of-flow in
        # test_liouville_3d, the 2D-reduction bridge in test_liouville_3d_2d_bridge,
        # and the C-vs-Python dxdv check in test_dxdv_3d_c_vs_python) all run over the
        # FULL registry, so adding a future potential is a one-line append below.
        # NB: future Pvar-pot families (e.g. additional non-axisymmetric potentials
        # that gain a full 3D C Hessian incl. zphideriv) append their potentials here.
        liouville3d_registry = [
            (
                potential.MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.1, normalize=True),
                "MiyamotoNagaiPotential",
                "axisymmetric",
            ),
            # a==0 exercises the disk->spherical special branch of the C Hessian
            (
                potential.MiyamotoNagaiPotential(amp=1.0, a=0.0, b=0.3, normalize=True),
                "MiyamotoNagaiPotential_a0",
                "axisymmetric",
            ),
            # MN3 expands to three MiyamotoNagai disks in C; exercises that the 3D
            # Hessian is correctly summed over the expanded components.
            (
                potential.MN3ExponentialDiskPotential(
                    amp=1.0, hr=1.0, hz=0.3, normalize=True
                ),
                "MN3ExponentialDiskPotential",
                "axisymmetric",
            ),
            # NOTE: KuzminDiskPotential has a verified-correct full 3D C Hessian
            # (hasC_dxdv3d=True), but it is intentionally NOT in this registry: its
            # potential ~ (a+|z|) is only C^0 across the disk plane, so d2Phi/dz2 and
            # d2Phi/dRdz are discontinuous at z=0. The registry's fixed IC crosses z=0,
            # where the two adaptive integrators legitimately diverge (~4e-6) at the
            # kink -- not a Hessian error. Off-plane the C vs Python dxdv agree to ~1e-11;
            # this is covered by test_orbit.test_kuzmindisk_dxdv_3d_c_vs_python_offplane.
            (
                potential.KuzminKutuzovStaeckelPotential(
                    amp=1.0, ac=5.0, Delta=1.0, normalize=True
                ),
                "KuzminKutuzovStaeckelPotential",
                "axisymmetric",
            ),
            (
                potential.FlattenedPowerPotential(
                    amp=1.0, alpha=0.5, q=0.9, normalize=True
                ),
                "FlattenedPowerPotential",
                "axisymmetric",
            ),
            # alpha==0 exercises the log-potential (LogarithmicHalo-like) branch of
            # the C Hessian (the alpha!=0 power-law branch is the default above).
            (
                potential.FlattenedPowerPotential(
                    amp=1.0, alpha=0.0, q=0.8, normalize=True
                ),
                "FlattenedPowerPotential_alpha0",
                "axisymmetric",
            ),
            # NOTE: DoubleExponentialDiskPotential has a verified-correct full 3D C
            # Hessian (hasC_dxdv3d=True) -- its C R2deriv/z2deriv/Rzderiv reproduce the
            # pure-Python reference dxdv to ~1e-9 -- but it is intentionally NOT in this
            # strict registry. Its forces (and 2nd derivatives) are evaluated by an
            # Ogata/Hankel Bessel quadrature, whose finite absolute accuracy means the
            # registry's finite-difference-of-the-flow check (eps=1e-7 differencing of
            # full nonlinear orbits) sits right at the ~1.2e-4 floor (just over the 1e-4
            # bound) -- NOT a Hessian error (the C-vs-Python dxdv gate passes at ~1e-9).
            # This follows the KuzminDisk/Einasto precedent: covered instead by the
            # dedicated test_orbit.test_doubleexp_dxdv_3d_c_vs_python.
            (
                potential.PlummerPotential(amp=1.0, b=0.7, normalize=True),
                "PlummerPotential",
                "spherical",
            ),
            (
                potential.HernquistPotential(amp=1.0, a=1.3, normalize=True),
                "HernquistPotential",
                "spherical",
            ),
            (
                potential.NFWPotential(amp=1.0, a=2.1, normalize=True),
                "NFWPotential",
                "spherical",
            ),
            (
                potential.JaffePotential(amp=1.0, a=1.7, normalize=True),
                "JaffePotential",
                "spherical",
            ),
            (
                potential.PowerSphericalPotential(amp=1.0, alpha=1.8, normalize=True),
                "PowerSphericalPotential",
                "spherical",
            ),
            (
                potential.PowerSphericalPotentialwCutoff(
                    amp=1.0, alpha=1.0, rc=2.0, normalize=True
                ),
                "PowerSphericalPotentialwCutoff",
                "spherical",
            ),
            (
                potential.DehnenSphericalPotential(
                    amp=1.0, a=1.5, alpha=1.5, normalize=True
                ),
                "DehnenSphericalPotential",
                "spherical",
            ),
            (
                potential.DehnenCoreSphericalPotential(amp=1.0, a=1.6, normalize=True),
                "DehnenCoreSphericalPotential",
                "spherical",
            ),
            (
                potential.BurkertPotential(amp=1.0, a=1.0, normalize=True),
                "BurkertPotential",
                "spherical",
            ),
            (
                potential.IsochronePotential(amp=1.0, b=1.2, normalize=True),
                "IsochronePotential",
                "spherical",
            ),
            (
                potential.HomogeneousSpherePotential(amp=1.0, R=3.0, normalize=True),
                "HomogeneousSpherePotential",
                "spherical",
            ),
            (
                potential.TwoPowerSphericalPotential(
                    amp=1.0, a=1.4, alpha=1.0, beta=4.0, normalize=True
                ),
                "TwoPowerSphericalPotential",
                "spherical",
            ),
            # NOTE: PseudoIsothermalPotential, EinastoPotential, and
            # interpSphericalPotential all have verified-correct full 3D C Hessians
            # (hasC_dxdv3d=True; their C-vs-Python unit-deviation dxdv agrees to ~1e-7
            # for every C integrator -- see test_spherical_dxdv_3d_c_vs_python_extra).
            # They are intentionally NOT in this strict registry, which also runs the
            # pure-Python odeint integrator and a 1e-9 3D->2D bridge tolerance:
            #  - Einasto and interpSpherical are spline-interpolated, so the loose
            #    odeint finite-difference-of-flow check (~1e-2 / and the unit-deviation
            #    bridge ~5e-9) is limited by the interpolation accuracy, not the Hessian.
            #  - PseudoIsothermal's (1/r^2)*atan(r/a) profile makes only the odeint
            #    FD-of-flow check marginally exceed 1e-4 (~1.8e-4) at the registry IC,
            #    while every C integrator agrees to ~1.6e-7.
            # NOTE: interpRZPotential also has a verified-correct full 3D C Hessian
            # (hasC_dxdv3d=True when the potential, forces, AND the three 2nd
            # derivatives are interpolated with enable_c; every C integrator matches
            # the pure-Python analytic dxdv of the UNDERLYING potential to ~1.0e-4,
            # interpolation-limited). It is intentionally NOT in this strict registry
            # (the interpSphericalPotential precedent): all its checks sit at spline
            # accuracy rather than the registry's analytic tolerances, and its
            # pure-Python integrator path with enable_c re-packs the full
            # interpolation grids into C per RHS evaluation, far too slow for the
            # registry sweep. Covered by the dedicated
            # test_orbit.test_interprz_dxdv_3d (C-vs-Python-on-underlying-potential,
            # det(M)=1/symplecticity, FD-of-flow).
            (
                potential.SpiralArmsPotential(),
                "SpiralArmsPotential",
                "nonaxisymmetric",
            ),
            # triaxial (b!=1) -> isNonAxi, exercises the full non-axi C Hessian
            # incl. zphideriv (nonzero off-plane for z!=0, phi!=0)
            (
                potential.LogarithmicHaloPotential(
                    amp=1.0, core=0.5, q=0.8, b=0.7, normalize=True
                ),
                "LogarithmicHaloPotential_triaxial",
                "nonaxisymmetric",
            ),
            # axisymmetric (b=None) -> the C Hessian's faster onem1overb2>=1
            # branch (no sin(phi) term), which the triaxial entry above never
            # exercises; covers those else-branches of R2/z2/Rz/phi2/Rphi/zphi.
            (
                potential.LogarithmicHaloPotential(
                    amp=1.0, core=0.5, q=0.8, normalize=True
                ),
                "LogarithmicHaloPotential_axi",
                "axisymmetric",
            ),
            # EllipsoidalPotential family: full 3D C Hessian via the Gauss-Legendre
            # angle integral over the ellipsoidal density. An oblate (b==1) instance
            # exercises the axisymmetric path; the triaxial (b!=1) instances exercise
            # the genuine non-axisymmetric path (nonzero zphideriv along the orbit).
            (
                potential.PerfectEllipsoidPotential(
                    amp=1.0, a=1.0, b=1.0, c=0.7, normalize=True
                ),
                "PerfectEllipsoidPotential_oblate",
                "axisymmetric",
            ),
            (
                potential.PerfectEllipsoidPotential(
                    amp=1.0, a=1.0, b=0.8, c=0.6, normalize=True
                ),
                "PerfectEllipsoidPotential_triaxial",
                "nonaxisymmetric",
            ),
            (
                potential.TriaxialNFWPotential(
                    amp=1.0, a=2.0, b=0.8, c=0.6, normalize=True
                ),
                "TriaxialNFWPotential",
                "nonaxisymmetric",
            ),
            (
                potential.TriaxialHernquistPotential(
                    amp=1.0, a=1.5, b=0.9, c=0.6, normalize=True
                ),
                "TriaxialHernquistPotential",
                "nonaxisymmetric",
            ),
            (
                # larger scale radius a keeps the fixed IC away from the steep
                # ~1/m central cusp, where the pure-Python odeint reference
                # integrator's finite-difference-of-flow check is otherwise noisy
                # (an integrator/FD-accuracy effect, not a Hessian error: the C
                # Hessian matches Python to ~1e-10 regardless, see
                # test_dxdv_3d_c_vs_python)
                potential.TriaxialJaffePotential(
                    amp=1.0, a=5.0, b=0.9, c=0.6, normalize=True
                ),
                "TriaxialJaffePotential",
                "nonaxisymmetric",
            ),
            (
                potential.TwoPowerTriaxialPotential(
                    amp=1.0, a=1.5, alpha=1.0, beta=4.0, b=0.8, c=0.6, normalize=True
                ),
                "TwoPowerTriaxialPotential",
                "nonaxisymmetric",
            ),
            (
                potential.TriaxialGaussianPotential(
                    amp=1.0, sigma=1.0, b=0.8, c=0.6, normalize=True
                ),
                "TriaxialGaussianPotential",
                "nonaxisymmetric",
            ),
            (
                potential.PowerTriaxialPotential(
                    amp=1.0, alpha=1.0, r1=1.0, b=0.8, c=0.6, normalize=True
                ),
                "PowerTriaxialPotential",
                "nonaxisymmetric",
            ),
            # Time-dependent, non-axisymmetric 3D bar: tform=-4 (in bar periods)
            # keeps the smoothing prefactor at 1 over the test interval, so the
            # full cos/sin(2(phi-Omega_b t-barphi)) angular dependence (incl. a
            # nonzero zphideriv off-plane) is exercised. alpha=0.05 (a standard
            # bar strength) raises |d2Phi/dz/dphi| along the fixed IC above the
            # 1e-3 guard so the C zphideriv coupling is genuinely tested.
            (
                potential.DehnenBarPotential(alpha=0.05),
                "DehnenBarPotential",
                "nonaxisymmetric",
            ),
            # ---- WrapperPotentials (Pvar-pot.6): the wrapper's full 3D C Hessian
            # is modulation x calc<deriv>(wrapped), so it is complete iff the
            # wrapped potential's 3D Hessian is in C (hasC_dxdv3d). Each wraps a
            # triaxial LogarithmicHalo so the genuine non-axisymmetric zphideriv
            # coupling is exercised; the smooth, time-dependent modulations keep
            # det(M)=1 / symplecticity (Hamiltonian flow) while making the
            # modulation factor non-trivial (!= 1) over the test interval. The
            # flow-direction identity (check 3) is auto-skipped for these
            # explicitly time-dependent potentials.
            (
                potential.DehnenSmoothWrapperPotential(
                    pot=potential.LogarithmicHaloPotential(
                        amp=1.0, core=0.5, q=0.8, b=0.7, normalize=True
                    ),
                    tform=-2.0,
                    tsteady=8.0,
                ),
                "DehnenSmoothWrapperPotential",
                "nonaxisymmetric",
            ),
            (
                potential.GaussianAmplitudeWrapperPotential(
                    pot=potential.LogarithmicHaloPotential(
                        amp=1.0, core=0.5, q=0.8, b=0.7, normalize=True
                    ),
                    to=2.5,
                    sigma=2.0,
                ),
                "GaussianAmplitudeWrapperPotential",
                "nonaxisymmetric",
            ),
            (
                potential.TimeDependentAmplitudeWrapperPotential(
                    pot=potential.LogarithmicHaloPotential(
                        amp=1.0, core=0.5, q=0.8, b=0.7, normalize=True
                    ),
                    A=_liouville3d_tdep_amp,
                ),
                "TimeDependentAmplitudeWrapperPotential",
                "nonaxisymmetric",
            ),
            # SolidBodyRotation only does something to an axisymmetric child
            # (phi -> phi - Omega t - pa is invisible to it), so wrapping the
            # triaxial Log makes the rotating-frame phi-shift -- and hence the
            # zphideriv coupling -- genuinely non-trivial.
            (
                potential.SolidBodyRotationWrapperPotential(
                    pot=potential.LogarithmicHaloPotential(
                        amp=1.0, core=0.5, q=0.8, b=0.7, normalize=True
                    ),
                    omega=1.3,
                    pa=0.2,
                ),
                "SolidBodyRotationWrapperPotential",
                "nonaxisymmetric",
            ),
            # ---- RotateAndTiltWrapperPotential: the wrapper's full 3D C Hessian
            # evaluates the wrapped potential's cylindrical Hessian at the
            # rotated (and optionally offset) point, builds the Cartesian Hessian
            # there, and conjugates back with the rotation matrix
            # (H = rot^T H' rot). Tilting breaks the z -> -z symmetry, so the
            # z=0 plane is NOT invariant and the 3D->2D bridge check is
            # auto-skipped for these entries (see _planar_invariant in
            # test_orbit.py). Three entries cover the C branch combinations:
            # rotation only, rotation+offset, and offset only (rotSet=false).
            (
                potential.RotateAndTiltWrapperPotential(
                    pot=potential.TriaxialNFWPotential(
                        amp=1.0, a=2.0, b=0.8, c=0.6, normalize=True
                    ),
                    galaxy_pa=0.3,
                    zvec=[numpy.sin(0.4), 0.0, numpy.cos(0.4)],
                ),
                "RotateAndTiltWrapperPotential_tiltedTriaxialNFW",
                "nonaxisymmetric",
            ),
            # inclination/sky_pa angle parametrization + offset: exercises the
            # offsetSet branch of the C Hessian (and the offset force path).
            # The offset is kept small because the default-tolerance pure-Python
            # odeint base orbit of the flow-direction check in test_liouville_3d
            # is otherwise marginally too inaccurate at the fixed registry IC
            # (an integrator-accuracy effect, NOT a Hessian error: the C
            # integrators pass regardless and the C Hessian matches the
            # pure-Python reference to ~4e-11, see test_dxdv_3d_c_vs_python);
            # the larger-offset C paths are covered by the norot entry below.
            (
                potential.RotateAndTiltWrapperPotential(
                    pot=potential.LogarithmicHaloPotential(
                        amp=1.0, core=0.5, q=0.8, b=0.7, normalize=True
                    ),
                    inclination=0.4,
                    galaxy_pa=0.3,
                    sky_pa=0.2,
                    offset=[0.03, -0.04, 0.02],
                ),
                "RotateAndTiltWrapperPotential_offset",
                "nonaxisymmetric",
            ),
            # offset WITHOUT rotation (norot): exercises the rotSet=false branch
            # of the C Hessian (no conjugation, offset-only point transform)
            (
                potential.RotateAndTiltWrapperPotential(
                    pot=potential.LogarithmicHaloPotential(
                        amp=1.0, core=0.5, q=0.8, b=0.7, normalize=True
                    ),
                    offset=[0.1, -0.15, 0.07],
                ),
                "RotateAndTiltWrapperPotential_norot_offset",
                "nonaxisymmetric",
            ),
        ]
        ids = [entry[1] for entry in liouville3d_registry]
        if metafunc.function.__name__ == "test_dxdv_3d_c_vs_python":
            # This test also wants the category, to switch on the non-axi check.
            metafunc.parametrize(
                "pot,pot_category",
                [(entry[0], entry[2]) for entry in liouville3d_registry],
                ids=ids,
            )
        else:
            # The det(M)/symplecticity/flow/FD-of-flow and 2D-bridge tests only need
            # the potential instance (the historical `pot` argument name).
            metafunc.parametrize(
                "pot",
                [entry[0] for entry in liouville3d_registry],
                ids=ids,
            )
    if metafunc.function.__name__ == "test_energy_jacobi_conservation":
        # Generate orbit integration tests for all potentials
        # Grab all of the potentials
        pots = [
            p
            for p in dir(potential)
            if (
                "Potential" in p
                and not "plot" in p
                and not "RZTo" in p
                and not "FullTo" in p
                and not "toPlanar" in p
                and not "evaluate" in p
                and not "Wrapper" in p
                and not "toVertical" in p
            )
        ]
        pots.append("mockFlatEllipticalDiskPotential")
        pots.append("mockFlatLopsidedDiskPotential")
        pots.append("mockFlatCosmphiDiskPotential")
        pots.append("mockFlatCosmphiDiskwBreakPotential")
        pots.append("mockSlowFlatEllipticalDiskPotential")
        pots.append("mockFlatDehnenBarPotential")
        pots.append("mockSlowFlatDehnenBarPotential")
        pots.append("mockFlatSteadyLogSpiralPotential")
        pots.append("mockSlowFlatSteadyLogSpiralPotential")
        pots.append("mockFlatTransientLogSpiralPotential")
        pots.append("specialMiyamotoNagaiPotential")
        pots.append("specialFlattenedPowerPotential")
        pots.append("BurkertPotentialNoC")
        pots.append("testMWPotential")
        pots.append("testplanarMWPotential")
        pots.append("mockMovingObjectLongIntPotential")
        pots.append("oblateHernquistPotential")
        pots.append("oblateNFWPotential")
        pots.append("prolateNFWPotential")
        pots.append("prolateJaffePotential")
        pots.append("triaxialNFWPotential")
        pots.append("fullyRotatedTriaxialNFWPotential")
        pots.append("NFWTwoPowerTriaxialPotential")  # for planar-from-full
        pots.append("mockSCFZeeuwPotential")
        pots.append("mockSCFNFWPotential")
        pots.append("mockSCFAxiDensity1Potential")
        pots.append("mockSCFAxiDensity2Potential")
        pots.append("mockSCFDensityPotential")
        pots.append("sech2DiskSCFPotential")
        pots.append("expwholeDiskSCFPotential")
        pots.append("altExpwholeDiskSCFPotential")
        pots.append("sech2DiskMultipoleExpansionPotential")
        pots.append("expwholeDiskMultipoleExpansionPotential")
        pots.append("mockFlatSpiralArmsPotential")
        pots.append("mockRotatingFlatSpiralArmsPotential")
        pots.append("mockSpecialRotatingFlatSpiralArmsPotential")
        pots.append("mockFlatDehnenSmoothBarPotential")
        pots.append("mockSlowFlatDehnenSmoothBarPotential")
        pots.append("mockSlowFlatDecayingDehnenSmoothBarPotential")
        pots.append("mockFlatSolidBodyRotationSpiralArmsPotential")
        pots.append("mockFlatSolidBodyRotationPlanarSpiralArmsPotential")
        pots.append("triaxialLogarithmicHaloPotential")
        pots.append("testorbitHenonHeilesPotential")
        pots.append("KuzminKutuzovOblateStaeckelWrapperPotential")
        pots.append("mockFlatCorotatingRotationSpiralArmsPotential")
        pots.append("mockFlatGaussianAmplitudeBarPotential")
        pots.append("nestedListPotential")
        pots.append("mockInterpSphericalPotential")
        pots.append("mockAdiabaticContractionMWP14WrapperPotential")
        pots.append("mockRotatedAndTiltedMWP14WrapperPotential")
        pots.append("testNullPotential")
        pots.append("mockKuzminLikeWrapperPotential")
        pots.append("MWP14CylindricallySeparablePotentialWrapper")
        pots.append("mockMultipoleExpansionSphericalPotential")
        pots.append("mockMultipoleExpansionAxiPotential")
        pots.append("mockMultipoleExpansionPotential")
        pots.append("mockFlatSolidBodyRotationMultipoleExpansionPotential")
        pots.append("mockFlatWeaklyTDMultipoleExpansionPotential")
        pots.append("mockFlatWeaklyTDNonaxiM3MultipoleExpansionPotential")
        rmpots = [
            "Potential",
            "MWPotential",
            "MWPotential2014",
            "MovingObjectPotential",
            "interpRZPotential",
            "linearPotential",
            "planarAxiPotential",
            "planarPotential",
            "verticalPotential",
            "PotentialError",
            "SnapshotRZPotential",
            "InterpSnapshotRZPotential",
            "EllipsoidalPotential",
            "NumericalPotentialDerivativesMixin",
            "SphericalHarmonicPotentialMixin",
            "SphericalPotential",
            "interpSphericalPotential",
            "CompositePotential",
            "planarCompositePotential",
            "baseCompositePotential",
            "KuijkenDubinskiDiskExpansionPotential",
        ]
        rmpots.append("SphericalShellPotential")
        rmpots.append("RingPotential")
        for p in rmpots:
            pots.remove(p)
        # tolerances in log10
        tol = {}
        tol["default"] = -10.0
        tol["DoubleExponentialDiskPotential"] = -6.0  # these are more difficult
        jactol = {}
        jactol["default"] = -10.0
        jactol["RazorThinExponentialDiskPotential"] = -9.0  # these are more difficult
        jactol["DoubleExponentialDiskPotential"] = -6.0  # these are more difficult
        jactol["mockFlatDehnenBarPotential"] = -8.0  # these are more difficult
        jactol["mockFlatDehnenSmoothBarPotential"] = -8.0  # these are more difficult
        jactol["mockMovingObjectLongIntPotential"] = -8.0  # these are more difficult
        jactol[
            "mockSlowFlatEllipticalDiskPotential"
        ] = -6.0  # these are more difficult (and also not quite conserved)
        jactol[
            "mockSlowFlatSteadyLogSpiralPotential"
        ] = -8.0  # these are more difficult (and also not quite conserved)
        jactol[
            "mockSlowFlatDehnenSmoothBarPotential"
        ] = -8.0  # these are more difficult (and also not quite conserved)
        jactol[
            "mockSlowFlatDecayingDehnenSmoothBarPotential"
        ] = -8.0  # these are more difficult (and also not quite conserved)
        jactol[
            "mockFlatSolidBodyRotationMultipoleExpansionPotential"
        ] = -4.0  # time-dependent, C integration
        jactol[
            "mockFlatWeaklyTDNonaxiM3MultipoleExpansionPotential"
        ] = -6.0  # time-dependent non-axi M=3, C integration
        # Now generate all inputs and run tests
        tols = [tol[p] if p in tol else tol["default"] for p in pots]
        jactols = [jactol[p] if p in jactol else tol["default"] for p in pots]
        firstTest = [True if ii == 0 else False for ii in range(len(pots))]
        metafunc.parametrize(
            "pot,ttol,tjactol,firstTest", zip(pots, tols, jactols, firstTest)
        )
    elif metafunc.function.__name__ == "test_energy_conservation_linear":
        # Generate linear orbit integration tests for all potentials
        # Grab all of the potentials
        pots = [
            p
            for p in dir(potential)
            if (
                "Potential" in p
                and not "plot" in p
                and not "RZTo" in p
                and not "FullTo" in p
                and not "toPlanar" in p
                and not "evaluate" in p
                and not "Wrapper" in p
                and not "toVertical" in p
            )
        ]
        pots.append("specialMiyamotoNagaiPotential")
        pots.append("specialFlattenedPowerPotential")
        pots.append("BurkertPotentialNoC")
        pots.append("testMWPotential")
        pots.append("testplanarMWPotential")
        pots.append("testlinearMWPotential")
        pots.append("mockCombLinearPotential")
        pots.append("mockSimpleLinearPotential")
        pots.append("oblateNFWPotential")
        pots.append("prolateNFWPotential")
        pots.append("triaxialNFWPotential")
        pots.append("fullyRotatedTriaxialNFWPotential")
        pots.append("NFWTwoPowerTriaxialPotential")  # for planar-from-full
        pots.append("mockSCFZeeuwPotential")
        pots.append("mockSCFNFWPotential")
        pots.append("mockSCFAxiDensity1Potential")
        pots.append("mockSCFAxiDensity2Potential")
        pots.append("sech2DiskSCFPotential")
        pots.append("expwholeDiskSCFPotential")
        pots.append("altExpwholeDiskSCFPotential")
        pots.append("sech2DiskMultipoleExpansionPotential")
        pots.append("expwholeDiskMultipoleExpansionPotential")
        pots.append("triaxialLogarithmicHaloPotential")
        pots.append("nestedListPotential")
        pots.append("mockInterpSphericalPotential")
        pots.append("mockAdiabaticContractionMWP14WrapperPotential")
        pots.append("mockRotatedAndTiltedMWP14WrapperPotential")
        pots.append("testNullPotential")
        pots.append("mockKuzminLikeWrapperPotential")
        pots.append("MWP14CylindricallySeparablePotentialWrapper")
        pots.append("mockMultipoleExpansionSphericalPotential")
        pots.append("mockMultipoleExpansionAxiPotential")
        rmpots = [
            "Potential",
            "MWPotential",
            "MWPotential2014",
            "MovingObjectPotential",
            "interpRZPotential",
            "linearPotential",
            "planarAxiPotential",
            "planarPotential",
            "verticalPotential",
            "PotentialError",
            "SnapshotRZPotential",
            "InterpSnapshotRZPotential",
            "EllipsoidalPotential",
            "NumericalPotentialDerivativesMixin",
            "SphericalHarmonicPotentialMixin",
            "SphericalPotential",
            "interpSphericalPotential",
            "CompositePotential",
            "planarCompositePotential",
            "baseCompositePotential",
            "KuijkenDubinskiDiskExpansionPotential",
        ]
        rmpots.append("SphericalShellPotential")
        rmpots.append("RingPotential")
        rmpots.append("SoftenedNeedleBarPotential")
        for p in rmpots:
            pots.remove(p)
        # tolerances in log10
        tol = {}
        tol["default"] = -10.0
        tol["DoubleExponentialDiskPotential"] = -6.0  # these are more difficult
        # Now generate all inputs and run tests
        tols = [tol[p] if p in tol else tol["default"] for p in pots]
        firstTest = [True if ii == 0 else False for ii in range(len(pots))]
        metafunc.parametrize("pot,ttol,firstTest", zip(pots, tols, firstTest))
    return None
