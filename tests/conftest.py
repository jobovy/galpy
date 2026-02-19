def pytest_generate_tests(metafunc):
    # galpy imports must be hear to not interfere with different config settings
    # in different files
    # Maybe I should define a cmdline option to set the config instead...
    from galpy import potential

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
