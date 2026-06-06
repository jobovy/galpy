def pytest_generate_tests(metafunc):
    # galpy imports must be hear to not interfere with different config settings
    # in different files
    # Maybe I should define a cmdline option to set the config instead...
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
                potential.SpiralArmsPotential(),
                "SpiralArmsPotential",
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
