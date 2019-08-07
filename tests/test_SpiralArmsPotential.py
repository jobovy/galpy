from __future__ import division
from galpy.potential import SpiralArmsPotential as spiral
import numpy as np
from numpy import pi
from numpy.testing import assert_allclose
from scipy.misc import derivative as deriv
import unittest

class TestSpiralArmsPotential(unittest.TestCase):

    def test_constructor(self):
        """Test that constructor initializes and converts units correctly."""
        sp = spiral()  # default values
        assert sp._amp == 1
        assert sp._N == -2  # trick to change to left handed coordinate system
        assert sp._alpha == -0.2
        assert sp._r_ref == 1
        assert sp._phi_ref == 0
        assert sp._Rs == 0.3
        assert sp._H == 0.125
        assert sp._Cs == [1]
        assert sp._omega == 0
        assert sp._rho0 == 1 / (4 * pi)
        assert sp.isNonAxi == True
        assert sp.hasC == True
        assert sp.hasC_dxdv == True
        assert sp._ro == 8
        assert sp._vo == 220

    def test_Rforce(self):
        """Tests Rforce against a numerical derivative -d(Potential) / dR."""
        dx = 1e-8
        rtol = 1e-5  # relative tolerance

        pot = spiral()
        assert_allclose(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx), rtol=rtol)
        R, z, t = 0.3, 0, 0
        assert_allclose(pot.Rforce(R, z, 0, t),        -deriv(lambda x: pot(x, z, 0, t),        R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2.2, t),   -deriv(lambda x: pot(x, z, pi/2.2, t),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi, t),       -deriv(lambda x: pot(x, z, pi, t),       R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3.7*pi/2, t), -deriv(lambda x: pot(x, z, 3.7*pi/2, t), R, dx=dx), rtol=rtol)
        R, z, t = 1, -.7, 3
        assert_allclose(pot.Rforce(R, z, 0, t),      -deriv(lambda x: pot(x, z, 0, t),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2, t), -deriv(lambda x: pot(x, z, pi/2, t), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi, t),     -deriv(lambda x: pot(x, z, pi, 0),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3.3*pi/2, t), -deriv(lambda x: pot(x, z, 3.3*pi/2, t), R, dx=dx), rtol=rtol)
        R, z = 3.14, .7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi),     -deriv(lambda x: pot(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)

        pot = spiral(amp=13, N=7, alpha=-0.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=3)
        assert_allclose(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(0.01, 0.), -deriv(lambda x: pot(x, 0.), 0.01, dx=dx), rtol=rtol)
        R, z, t = 0.3, 0, 1.123
        assert_allclose(pot.Rforce(R, z, 0, t),      -deriv(lambda x: pot(x, z, 0, t),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2, t),   -deriv(lambda x: pot(x, z, pi/2, t),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi, t),     -deriv(lambda x: pot(x, z, pi, t),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2, t), -deriv(lambda x: pot(x, z, 3*pi/2, t), R, dx=dx), rtol=rtol)
        R, z, t = 1, -.7, 121
        assert_allclose(pot.Rforce(R, z, 0, t),      -deriv(lambda x: pot(x, z, 0, t),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2, t), -deriv(lambda x: pot(x, z, pi / 2, t), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi, t),     -deriv(lambda x: pot(x, z, pi, t),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2, t), -deriv(lambda x: pot(x, z, 3*pi/2, t), R, dx=dx), rtol=rtol)
        R, z, t = 3.14, .7, 0.123
        assert_allclose(pot.Rforce(R, z, 0, t),      -deriv(lambda x: pot(x, z, 0, t),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2, t), -deriv(lambda x: pot(x, z, pi / 2, t), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi, t),     -deriv(lambda x: pot(x, z, pi, t),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2, t), -deriv(lambda x: pot(x, z, 3*pi/2, t), R, dx=dx), rtol=rtol)

        pot = spiral(amp=13, N=1, alpha=0.01, r_ref=1.12, phi_ref=0, Cs=[1, 1.5, 8.], omega=-3)
        assert_allclose(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(0.1, 0.), -deriv(lambda x: pot(x, 0.), 0.1, dx=dx), rtol=rtol)
        R, z, t = 0.3, 0, -4.5
        assert_allclose(pot.Rforce(R, z, 0, t),      -deriv(lambda x: pot(x, z, 0, t),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2, t),   -deriv(lambda x: pot(x, z, pi/2, t),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi, t),     -deriv(lambda x: pot(x, z, pi, t),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2, t), -deriv(lambda x: pot(x, z, 3*pi/2, t), R, dx=dx), rtol=rtol)
        R, z, t = 1, -.7, -123
        assert_allclose(pot.Rforce(R, z, 0, t),      -deriv(lambda x: pot(x, z, 0, t),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2, t), -deriv(lambda x: pot(x, z, pi / 2, t), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi, t),     -deriv(lambda x: pot(x, z, pi, t),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2, t), -deriv(lambda x: pot(x, z, 3*pi/2, t), R, dx=dx), rtol=rtol)
        R, z, t = 3.14, .7, -123.123
        assert_allclose(pot.Rforce(R, z, 0, t),      -deriv(lambda x: pot(x, z, 0, t),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2, t), -deriv(lambda x: pot(x, z, pi/2, t), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi, t),     -deriv(lambda x: pot(x, z, pi, t),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2, t), -deriv(lambda x: pot(x, z, 3*pi/2, t), R, dx=dx), rtol=rtol)

        pot = spiral(N=10, r_ref=15, phi_ref=5, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        assert_allclose(pot.Rforce(1., 0.), -deriv(lambda x: pot(x, 0.), 1., dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(0.01, 0.), -deriv(lambda x: pot(x, 0.), 0.01, dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi/2.1),   -deriv(lambda x: pot(x, z, pi/2.1),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 1.3*pi),     -deriv(lambda x: pot(x, z, 1.3*pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3*pi/2), -deriv(lambda x: pot(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2), -deriv(lambda x: pot(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, .9*pi),     -deriv(lambda x: pot(x, z, .9*pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3.3*pi/2), -deriv(lambda x: pot(x, z, 3.3*pi/2), R, dx=dx), rtol=rtol)
        R, z = 3.14, .7
        assert_allclose(pot.Rforce(R, z, 0),      -deriv(lambda x: pot(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, pi / 2.3), -deriv(lambda x: pot(x, z, pi / 2.3), R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 1.1*pi),     -deriv(lambda x: pot(x, z, 1.1*pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.Rforce(R, z, 3.5*pi/2), -deriv(lambda x: pot(x, z, 3.5*pi/2), R, dx=dx), rtol=rtol)

    def test_zforce(self):
        """Test zforce against a numerical derivative -d(Potential) / dz"""
        dx = 1e-8
        rtol = 1e-6  # relative tolerance

        pot = spiral()
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(0.3, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi/2), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, 3*pi/2), rtol=rtol)
        # test zforce against -dPhi/dz
        R, z = 1, -.7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 3.7, .7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral(amp=13, N=3, alpha=-.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2], omega=3)
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(0.3, 0, 0, 1), rtol=rtol)
        assert_allclose(0, pot.zforce(0.6, 0, pi/2, 2), rtol=rtol)
        assert_allclose(0, pot.zforce(0.9, 0, pi, 3), rtol=rtol)
        assert_allclose(0, pot.zforce(1.2, 0, 2*pi, 4), rtol=rtol)
        # test zforce against -dPhi/dz
        R, z, t = 1, -.7, 123
        assert_allclose(pot.zforce(R, z, 0, t),      -deriv(lambda x: pot(R, x, 0, t),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2, t),   -deriv(lambda x: pot(R, x, pi/2, t),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi, t),     -deriv(lambda x: pot(R, x, pi, t),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2, t), -deriv(lambda x: pot(R, x, 3*pi/2, t), z, dx=dx), rtol=rtol)
        R, z = 3.7, .7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral(N=1, alpha=-0.2, r_ref=.5, Cs=[1, 1.5], omega=-3)
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(0.3, 0, 0, 123), rtol=rtol)
        assert_allclose(0, pot.zforce(0.3, 0, pi/2, -321), rtol=rtol)
        assert_allclose(0, pot.zforce(32, 0, pi, 1.23), rtol=rtol)
        assert_allclose(0, pot.zforce(0.123, 0, 3.33*pi/2, -3.21), rtol=rtol)
        # test zforce against -dPhi/dz
        R, z = 1, -1.5
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2.1), -deriv(lambda x: pot(R, x, 3*pi/2.1), z, dx=dx), rtol=rtol)
        R, z, t = 3.7, .7, -100
        assert_allclose(pot.zforce(R, z, 0, t),      -deriv(lambda x: pot(R, x, 0, t),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2, t),   -deriv(lambda x: pot(R, x, pi/2, t),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi, t),     -deriv(lambda x: pot(R, x, pi, t),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3.4*pi/2, t), -deriv(lambda x: pot(R, x, 3.4*pi/2, t), z, dx=dx), rtol=rtol)

        pot = spiral(N=5, r_ref=1.5, phi_ref=0.5, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        # zforce is zero in the plane of the galaxy
        assert_allclose(0, pot.zforce(0.3, 0, 0), rtol=rtol)
        assert_allclose(0, pot.zforce(0.4, 0, pi/2), rtol=rtol)
        assert_allclose(0, pot.zforce(0.5, 0, pi*1.1), rtol=rtol)
        assert_allclose(0, pot.zforce(0.6, 0, 3*pi/2), rtol=rtol)
        # test zforce against -dPhi/dz
        R, z = 1, -.7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 37, 1.7
        assert_allclose(pot.zforce(R, z, 0),      -deriv(lambda x: pot(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi/2),   -deriv(lambda x: pot(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, pi),     -deriv(lambda x: pot(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.zforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

    def test_phiforce(self):
        """Test phiforce against a numerical derivative -d(Potential) / d(phi)."""
        dx = 1e-8
        rtol = 1e-5  # relative tolerance

        pot = spiral()
        R, z = .3, 0
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = .1, -.3
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3, 7
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2.1), -deriv(lambda x: pot(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

        pot = spiral(N=7, alpha=-0.3, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 1, 1], omega=2*pi)
        R, z, t = .3, 0, 1.2
        assert_allclose(pot.phiforce(R, z, 0, 0),      -deriv(lambda x: pot(R, z, x, 0),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2, t),   -deriv(lambda x: pot(R, z, x, t),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi, t),     -deriv(lambda x: pot(R, z, x, t),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2, t), -deriv(lambda x: pot(R, z, x, t), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z, t = 3.7, .7, -5.1
        assert_allclose(pot.phiforce(R, z, 0, t),      -deriv(lambda x: pot(R, z, x, t),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2, t),   -deriv(lambda x: pot(R, z, x, t),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi, t),     -deriv(lambda x: pot(R, z, x, t),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3.2*pi/2, t), -deriv(lambda x: pot(R, z, x, t), 3.2*pi/2, dx=dx), rtol=rtol)

        pot = spiral(N=1, alpha=0.1, phi_ref=0, Cs=[1, 1.5], omega=-.333)
        R, z = .3, 0
        assert_allclose(pot.phiforce(R, z, 0),        -deriv(lambda x: pot(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),     -deriv(lambda x: pot(R, z, x),   pi/2,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),       -deriv(lambda x: pot(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3.2*pi/2), -deriv(lambda x: pot(R, z, x), 3.2*pi/2, dx=dx), rtol=rtol)
        R, z, t = 1, -.7, 123
        assert_allclose(pot.phiforce(R, z, 0, t),      -deriv(lambda x: pot(R, z, x, t),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2, t),   -deriv(lambda x: pot(R, z, x, t),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi, t),     -deriv(lambda x: pot(R, z, x, t),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2, t), -deriv(lambda x: pot(R, z, x, t), 3*pi/2, dx=dx), rtol=rtol)
        R, z, t = 3, 4, 5
        assert_allclose(pot.phiforce(R, z, 0, t),      -deriv(lambda x: pot(R, z, x, t),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2, t),   -deriv(lambda x: pot(R, z, x, t),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi, t),     -deriv(lambda x: pot(R, z, x, t),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2, t), -deriv(lambda x: pot(R, z, x, t), 3*pi/2, dx=dx), rtol=rtol)

        pot = spiral(N=4, r_ref=1.5, phi_ref=5, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        R, z = .3, 0
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 3*pi/2), -deriv(lambda x: pot(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 2.1, .12345
        assert_allclose(pot.phiforce(R, z, 0),      -deriv(lambda x: pot(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi/2),   -deriv(lambda x: pot(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, pi),     -deriv(lambda x: pot(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phiforce(R, z, 2*pi), -deriv(lambda x: pot(R, z, x), 2*pi, dx=dx), rtol=rtol)

    def test_R2deriv(self):
        """Test R2deriv against a numerical derivative -d(Rforce) / dR."""
        dx = 1e-8
        rtol = 1e-6  # relative tolerance

        pot = spiral()
        assert_allclose(pot.R2deriv(1., 0.),           -deriv(lambda x: pot.Rforce(x, 0.), 1.,   dx=dx), rtol=rtol)
        R, z = 0.3, 0
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi/2),   R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3.1*pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3.1*pi/2), R, dx=dx), rtol=rtol)
        R, z = 1, -.7
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 2*pi), -deriv(lambda x: pot.Rforce(x, z, 2*pi), R, dx=dx), rtol=rtol)
        R, z = 5, .9
        assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)

        # pot = spiral(N=1, alpha=-.3, r_ref=.1, phi_ref=pi, Rs=1, H=1, Cs=[1, 2, 3], omega=3)
        # assert_allclose(pot.R2deriv(1e-3, 0.),         -deriv(lambda x: pot.Rforce(x, 0.), 1e-3, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(1., 0.),           -deriv(lambda x: pot.Rforce(x, 0.), 1.,   dx=dx), rtol=rtol)
        # R, z = 0.3, 0
        # assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi/2),   R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        # R, z = 1, -.7
        # assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, 3.1*pi/2), -deriv(lambda x: pot.Rforce(x, z, 3.1*pi/2), R, dx=dx), rtol=rtol)
        # R, z = 5, .9
        # assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi / 2.4),     -deriv(lambda x: pot.Rforce(x, z, pi / 2.4), R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        #
        # pot = spiral(N=7, alpha=.1, r_ref=1, phi_ref=1, Rs=1.1, H=.1, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)], omega=-.3)
        # assert_allclose(pot.R2deriv(1., 0.),           -deriv(lambda x: pot.Rforce(x, 0.), 1.,   dx=dx), rtol=rtol)
        # R, z = 0.3, 0
        # assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi/2),     -deriv(lambda x: pot.Rforce(x, z, pi/2),   R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        # R, z = 1, -.7
        # assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        # R, z = 5, .9
        # assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        #
        # pot = spiral(N=4, alpha=pi/2, r_ref=1, phi_ref=1, Rs=.7, H=.77, Cs=[3, 4], omega=-1.3)
        # assert_allclose(pot.R2deriv(1e-3, 0.),         -deriv(lambda x: pot.Rforce(x, 0.), 1e-3, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(1., 0.),           -deriv(lambda x: pot.Rforce(x, 0.), 1.,   dx=dx), rtol=rtol)
        # R, z = 0.3, 0
        # assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi/2),   R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, 3 * pi / 2), -deriv(lambda x: pot.Rforce(x, z, 3*pi/2), R, dx=dx), rtol=rtol)
        # R, z = 1, -.7
        # assert_allclose(pot.R2deriv(R, z, 0),          -deriv(lambda x: pot.Rforce(x, z, 0),      R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi / 2),     -deriv(lambda x: pot.Rforce(x, z, pi / 2), R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, pi),         -deriv(lambda x: pot.Rforce(x, z, pi),     R, dx=dx), rtol=rtol)
        # assert_allclose(pot.R2deriv(R, z, .33*pi/2), -deriv(lambda x: pot.Rforce(x, z, .33*pi/2), R, dx=dx), rtol=rtol)

    def test_z2deriv(self):
        """Test z2deriv against a numerical derivative -d(zforce) / dz"""
        dx = 1e-8
        rtol = 1e-6  # relative tolerance

        pot = spiral()
        R, z = .3, 0
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 1.2, .1
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral(N=3, alpha=-0.3, r_ref=.25, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        R, z = .3, 0
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral(amp=5, N=1, alpha=0.1, r_ref=0.5, phi_ref=0.3, Rs=0.7, H=0.7, Cs=[1, 2], omega=3)
        R, z = .3, 0
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

        pot = spiral(N=1, alpha=1, r_ref=3, phi_ref=pi, Cs=[1, 2], omega=-3)
        R, z = .7, 0
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)
        R, z = 2.1, .99
        assert_allclose(pot.z2deriv(R, z, 0),      -deriv(lambda x: pot.zforce(R, x, 0),      z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi/2),   -deriv(lambda x: pot.zforce(R, x, pi/2),   z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, pi),     -deriv(lambda x: pot.zforce(R, x, pi),     z, dx=dx), rtol=rtol)
        assert_allclose(pot.z2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.zforce(R, x, 3*pi/2), z, dx=dx), rtol=rtol)

    def test_phi2deriv(self):
        """Test phi2deriv against a numerical derivative -d(phiforce) / d(phi)."""
        dx = 1e-8
        rtol = 1e-7  # relative tolerance

        pot = spiral()
        R, z = .3, 0
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.1),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2.5), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2.5, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.1), -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

        pot = spiral(amp=13, N=1, alpha=-.3, r_ref=0.5, phi_ref=0.1, Rs=0.7, H=0.7, Cs=[1, 2, 3], omega=3)
        R, z = .3, 0
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3.3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3.3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.1), -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

        pot = spiral(amp=13, N=5, alpha=0.1, r_ref=.3, phi_ref=.1, Rs=0.77, H=0.747, Cs=[3, 2], omega=-3)
        R, z = .3, 0
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.3
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 3.3, .7
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.1), -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

        pot = spiral(amp=11, N=7, alpha=.777, r_ref=7, phi_ref=.7, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)])
        R, z = .7, 0
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1, -.33
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.2),   -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.2, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2, dx=dx), rtol=rtol)
        R, z = 1.123, .123
        assert_allclose(pot.phi2deriv(R, z, 0),      -deriv(lambda x: pot.phiforce(R, z, x),      0,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi/2.1), -deriv(lambda x: pot.phiforce(R, z, x),   pi/2.1, dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, pi),     -deriv(lambda x: pot.phiforce(R, z, x),     pi,   dx=dx), rtol=rtol)
        assert_allclose(pot.phi2deriv(R, z, 3*pi/2), -deriv(lambda x: pot.phiforce(R, z, x), 3*pi/2,   dx=dx), rtol=rtol)

    def test_dens(self):
        """Test dens against density obtained using Poisson's equation."""
        rtol = 1e-2  # relative tolerance (this one isn't as precise)

        pot = spiral()
        assert_allclose(pot.dens(1, 0, 0, forcepoisson=False), pot.dens(1, 0, 0, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, 1, .5, forcepoisson=False), pot.dens(1, 1, .5, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, -1, -1, forcepoisson=False), pot.dens(1, -1, -1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(.1, .1, .1, forcepoisson=False), pot.dens(.1, .1, .1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(33, .777, .747, forcepoisson=False), pot.dens(33, .777, .747, forcepoisson=True), rtol=rtol)

        pot = spiral(amp=3, N=5, alpha=.3, r_ref=.7, omega=5)
        assert_allclose(pot.dens(1, 0, 0, forcepoisson=False), pot.dens(1, 0, 0, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1.2, 1.2, 1.2, forcepoisson=False), pot.dens(1.2, 1.2, 1.2, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, -1, -1, forcepoisson=False), pot.dens(1, -1, -1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(.1, .1, .1, forcepoisson=False), pot.dens(.1, .1, .1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(33.3, .007, .747, forcepoisson=False), pot.dens(33.3, .007, .747, forcepoisson=True), rtol=rtol)

        pot = spiral(amp=0.6, N=3, alpha=.24, r_ref=1, phi_ref=pi, Cs=[8./(3.*pi), 0.5, 8./(15.*pi)], omega=-3)
        assert_allclose(pot.dens(1, 0, 0, forcepoisson=False), pot.dens(1, 0, 0, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, 1, 1, forcepoisson=False), pot.dens(1, 1, 1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, -1, -1, forcepoisson=False), pot.dens(1, -1, -1, forcepoisson=True), rtol=rtol)
#        assert_allclose(pot.dens(.1, .1, .1, forcepoisson=False), pot.dens(.1, .1, .1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(3.33, -7.77, -.747, forcepoisson=False), pot.dens(3.33, -7.77, -.747, forcepoisson=True), rtol=rtol)

        pot = spiral(amp=100, N=4, alpha=pi/2, r_ref=1, phi_ref=1, Rs=7, H=77, Cs=[3, 1, 1], omega=-1.3)
        assert_allclose(pot.dens(1, 0, 0, forcepoisson=False), pot.dens(1, 0, 0, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(3, 2, pi, forcepoisson=False), pot.dens(3, 2, pi, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(1, -1, -1, forcepoisson=False), pot.dens(1, -1, -1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(.1, .123, .1, forcepoisson=False), pot.dens(.1, .123, .1, forcepoisson=True), rtol=rtol)
        assert_allclose(pot.dens(333, -.777, .747, forcepoisson=False), pot.dens(333, -.777, .747, forcepoisson=True), rtol=rtol)

    def test_Rzderiv(self):
        """Test Rzderiv against a numerical derivative."""
        dx = 1e-8
        rtol = 1e-6

        pot = spiral()
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 0.7, 0.3, pi/3, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.1, -0.3, pi/4.2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 12, 1, 2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 5, -.7, 3*pi/2, 5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 4, 7, 2, 10000
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.23, 0, 44, 343
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)

        pot = spiral(amp=13, N=7, alpha=.1, r_ref=1.123, phi_ref=.3, Rs=0.777, H=.5, Cs=[4.5], omega=-3.4)
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .777, 0.333, pi/3, 0.
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.1, -0.3, pi/4.2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 12, 1, 2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 2, -.7, 3*pi/2, 5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 4, 7, 2, 10000
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.23, 0, 44, 343
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)

        pot = spiral(amp=11, N=2, alpha=.777, r_ref=7, Cs=[8.], omega=0.1)
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 0.7, 0.3, pi/12, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.1, -0.3, pi/4.2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 2, 1, 2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 5, -.7, 3*pi/2, 5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 4, 7, 2, 10000
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.23, 0, 44, 343
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)

        pot = spiral(amp=2, N=1, alpha=-0.1, r_ref=5, Rs=5, H=.7, Cs=[3.5], omega=3)
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 0.77, 0.3, pi/3, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 3.1, -0.3, pi/5, 2
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 12, 1, 2, 3
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 5, -.7, 3*pi/2, 5
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 4, 7, 2, 10000
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 1.23, 0, 44, 343
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rzderiv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx), rtol=rtol)

    def test_Rphideriv(self):
        """Test Rphideriv against a numerical derivative."""
        dx = 1e-8
        rtol = 5e-5

        pot = spiral()
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 0.7, 0.3, pi / 3, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 1.1, -0.3, pi / 4.2, 3
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 12, 1, 2, 3
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 5, -.7, 3 * pi / 2, 5
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 4, 7, 2, 1000
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 1.23, 0, 44, 343
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 1, 7, 7
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)

        pot = spiral(N=3, alpha=.21, r_ref=.5, phi_ref=pi, Cs=[2.], omega=-3)
        R, z, phi, t = 1, 0, 0, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 0.7, 0.3, pi / 3, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 1.1, -0.3, pi / 4.2, 3
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = .777, .747, .343, 2.5
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 12, 1, 2, 3
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 4, 5, 6
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 5, -.7, 3 * pi / 2, 5
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 11, 11, 11, 1.123
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 3, 2, 1, 100
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = .01, 0, 0, 0
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 1.12, 0, 2, 343
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)
        R, z, phi, t = 7, 7, 7, 7
        assert_allclose(pot.Rphideriv(R, z, phi, t), -deriv(lambda x: pot.Rforce(R, z, x, t), phi, dx=dx), rtol=rtol)

    def test_OmegaP(self):
        sp = spiral()
        assert sp.OmegaP() == 0

        sp = spiral(N=1, alpha=2, r_ref=.1, phi_ref=.5, Rs=0.2, H=0.7, Cs=[1,2], omega=-123)
        assert sp.OmegaP() == -123

        sp = spiral(omega=123.456)
        assert sp.OmegaP() == 123.456

    def test_K(self):
        pot = spiral()
        R = 1
        assert_allclose([pot._K(R)], [pot._ns * pot._N / R / np.sin(pot._alpha)])

        R = 1e-6
        assert_allclose([pot._K(R)], [pot._ns * pot._N / R / np.sin(pot._alpha)])

        R = 0.5
        assert_allclose([pot._K(R)], [pot._ns * pot._N / R / np.sin(pot._alpha)])

    def test_B(self):
        pot = spiral()

        R = 1
        assert_allclose([pot._B(R)], [pot._K(R) * pot._H * (1 + 0.4 * pot._K(R) * pot._H)])

        R = 1e-6
        assert_allclose([pot._B(R)], [pot._K(R) * pot._H * (1 + 0.4 * pot._K(R) * pot._H)])

        R = 0.3
        assert_allclose([pot._B(R)], [pot._K(R) * pot._H * (1 + 0.4 * pot._K(R) * pot._H)])

    def test_D(self):
        pot = spiral()

        assert_allclose([pot._D(3)], [(1. + pot._K(3)*pot._H + 0.3 * pot._K(3)**2 * pot._H**2.) / (1. + 0.3*pot._K(3) * pot._H)])
        assert_allclose([pot._D(1e-6)], [(1. + pot._K(1e-6)*pot._H + 0.3 * pot._K(1e-6)**2 * pot._H**2.) / (1. + 0.3*pot._K(1e-6) * pot._H)])
        assert_allclose([pot._D(.5)], [(1. + pot._K(.5)*pot._H + 0.3 * pot._K(.5)**2 * pot._H**2.) / (1. + 0.3*pot._K(.5) * pot._H)])

    def test_dK_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dK_dR(3), deriv(pot._K, 3, dx=dx))
        assert_allclose(pot._dK_dR(2.3), deriv(pot._K, 2.3, dx=dx))
        assert_allclose(pot._dK_dR(-2.3), deriv(pot._K, -2.3, dx=dx))

    def test_dB_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dB_dR(3.3), deriv(pot._B, 3.3, dx=dx))
        assert_allclose(pot._dB_dR(1e-3), deriv(pot._B, 1e-3, dx=dx))
        assert_allclose(pot._dB_dR(3), deriv(pot._B, 3, dx=dx))

    def test_dD_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dD_dR(1e-3), deriv(pot._D, 1e-3, dx=dx))
        assert_allclose(pot._dD_dR(2), deriv(pot._D, 2, dx=dx))

    def test_gamma(self):
        pot = spiral()

        R, phi = 1, 2
        assert_allclose(pot._gamma(R, phi), [pot._N * (float(phi) - pot._phi_ref - np.log(float(R) / pot._r_ref) /
                                                         np.tan(pot._alpha))])

        R , phi = .1, -.2
        assert_allclose(pot._gamma(R, phi), [pot._N * (float(phi) - pot._phi_ref - np.log(float(R) / pot._r_ref) /
                                                         np.tan(pot._alpha))])

        R, phi = 0.01, 0
        assert_allclose(pot._gamma(R, phi), [pot._N * (float(phi) - pot._phi_ref - np.log(float(R) / pot._r_ref) /
                                                         np.tan(pot._alpha))])

    def test_dgamma_dR(self):
        pot = spiral()

        dx = 1e-8
        assert_allclose(pot._dgamma_dR(3.), deriv(lambda x: pot._gamma(x, 1), 3., dx=dx))
        assert_allclose(pot._dgamma_dR(3), deriv(lambda x: pot._gamma(x, 1), 3, dx=dx))
        assert_allclose(pot._dgamma_dR(0.01), deriv(lambda x: pot._gamma(x, 1), 0.01, dx=dx))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpiralArmsPotential)
    unittest.TextTestRunner(verbosity=2).run(suite)
