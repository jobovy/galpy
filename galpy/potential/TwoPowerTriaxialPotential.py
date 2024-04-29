###############################################################################
#   TwoPowerTriaxialPotential.py: General class for triaxial potentials
#                                 derived from densities with two power-laws
#
#                                                    amp/[4pia^3]
#                             rho(r)= ------------------------------------
#                                      (m/a)^\alpha (1+m/a)^(\beta-\alpha)
#
#                             with
#
#                             m^2 = x^2 + y^2/b^2 + z^2/c^2
###############################################################################
import numpy
from scipy import special

from ..util import conversion
from .EllipsoidalPotential import EllipsoidalPotential


class TwoPowerTriaxialPotential(EllipsoidalPotential):
    """Class that implements triaxial potentials that are derived from
    two-power density models

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)^\\alpha\\,(1+m/a)^{\\beta-\\alpha}}

    with

    .. math::

        m^2 = x'^2 + \\frac{y'^2}{b^2}+\\frac{z'^2}{c^2}

    and :math:`(x',y',z')` is a rotated frame wrt :math:`(x,y,z)` specified by parameters ``zvec`` and ``pa`` which specify (a) ``zvec``: the location of the :math:`z'` axis in the :math:`(x,y,z)` frame and (b) ``pa``: the position angle of the :math:`x'` axis wrt the :math:`\\tilde{x}` axis, that is, the :math:`x` axis after rotating to ``zvec``.

    Note that this general class of potentials does *not* automatically revert to the special TriaxialNFWPotential, TriaxialHernquistPotential, or TriaxialJaffePotential when using their (alpha,beta) values (like TwoPowerSphericalPotential).
    """

    def __init__(
        self,
        amp=1.0,
        a=5.0,
        alpha=1.5,
        beta=3.5,
        b=1.0,
        c=1.0,
        zvec=None,
        pa=None,
        glorder=50,
        normalize=False,
        ro=None,
        vo=None,
    ):
        """
        Initialize a triaxial two-power-density potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Scale radius.
        alpha : float, optional
            Inner power (0 <= alpha < 3).
        beta : float, optional
            Outer power ( beta > 2).
        b : float, optional
            y-to-x axis ratio of the density.
        c : float, optional
            z-to-x axis ratio of the density.
        zvec : numpy.ndarray, optional
            If set, a unit vector that corresponds to the z axis.
        pa : float or Quantity, optional
            If set, the position angle of the x axis.
        glorder : int, optional
            If set, compute the relevant force and potential integrals with Gaussian quadrature of this order.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2016-05-30 - Started - Bovy (UofT)
        - 2018-08-07 - Re-written using the general EllipsoidalPotential class - Bovy (UofT)

        """
        EllipsoidalPotential.__init__(
            self,
            amp=amp,
            b=b,
            c=c,
            zvec=zvec,
            pa=pa,
            glorder=glorder,
            ro=ro,
            vo=vo,
            amp_units="mass",
        )
        a = conversion.parse_length(a, ro=self._ro)
        self.a = a
        self._scale = self.a
        if beta <= 2.0 or alpha >= 3.0:
            raise OSError(
                "TwoPowerTriaxialPotential requires 0 <= alpha < 3 and beta > 2"
            )
        self.alpha = alpha
        self.beta = beta
        self.betaminusalpha = self.beta - self.alpha
        self.twominusalpha = 2.0 - self.alpha
        self.threeminusalpha = 3.0 - self.alpha
        if self.twominusalpha != 0.0:
            self.psi_inf = (
                special.gamma(self.beta - 2.0)
                * special.gamma(3.0 - self.alpha)
                / special.gamma(self.betaminusalpha)
            )
        # Adjust amp
        self._amp /= 4.0 * numpy.pi * self.a**3
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        return None

    def _psi(self, m):
        r"""\psi(m) = -\int_{m^2}^\infty d m'^2 \rho(m'^2)"""
        if self.twominusalpha == 0.0:
            return (
                -2.0
                * self.a**2
                * (self.a / m) ** self.betaminusalpha
                / self.betaminusalpha
                * special.hyp2f1(
                    self.betaminusalpha,
                    self.betaminusalpha,
                    self.betaminusalpha + 1,
                    -self.a / m,
                )
            )
        else:
            return (
                -2.0
                * self.a**2
                * (
                    self.psi_inf
                    - (m / self.a) ** self.twominusalpha
                    / self.twominusalpha
                    * special.hyp2f1(
                        self.twominusalpha,
                        self.betaminusalpha,
                        self.threeminusalpha,
                        -m / self.a,
                    )
                )
            )

    def _mdens(self, m):
        """Density as a function of m"""
        return (self.a / m) ** self.alpha / (1.0 + m / self.a) ** (self.betaminusalpha)

    def _mdens_deriv(self, m):
        """Derivative of the density as a function of m"""
        return (
            -self._mdens(m) * (self.a * self.alpha + self.beta * m) / m / (self.a + m)
        )

    def _mass(self, R, z=None, t=0.0):
        if not z is None:
            raise AttributeError  # Hack to fall back to general
        return (
            4.0
            * numpy.pi
            * self.a**self.alpha
            * R ** (3.0 - self.alpha)
            / (3.0 - self.alpha)
            * self._b
            * self._c
            * special.hyp2f1(
                3.0 - self.alpha, self.betaminusalpha, 4.0 - self.alpha, -R / self.a
            )
        )


class TriaxialHernquistPotential(EllipsoidalPotential):
    """Class that implements the triaxial Hernquist potential

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)\\,(1+m/a)^{3}}

    with

    .. math::

        m^2 = x'^2 + \\frac{y'^2}{b^2}+\\frac{z'^2}{c^2}

    and :math:`(x',y',z')` is a rotated frame wrt :math:`(x,y,z)` specified by parameters ``zvec`` and ``pa`` which specify (a) ``zvec``: the location of the :math:`z'` axis in the :math:`(x,y,z)` frame and (b) ``pa``: the position angle of the :math:`x'` axis wrt the :math:`\\tilde{x}` axis, that is, the :math:`x` axis after rotating to ``zvec``.

    """

    def __init__(
        self,
        amp=1.0,
        a=2.0,
        normalize=False,
        b=1.0,
        c=1.0,
        zvec=None,
        pa=None,
        glorder=50,
        ro=None,
        vo=None,
    ):
        """
        Initialize a triaxial two-power-density potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass.
        a : float or Quantity, optional
            Scale radius.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        b : float, optional
            y-to-x axis ratio of the density.
        c : float, optional
            z-to-x axis ratio of the density.
        zvec : numpy.ndarray, optional
            If set, a unit vector that corresponds to the z axis.
        pa : float or Quantity, optional
            If set, the position angle of the x axis.
        glorder : int, optional
            If set, compute the relevant force and potential integrals with Gaussian quadrature of this order.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2010-07-09 - Written - Bovy (UofT)

        - 2018-08-07 - Re-written using the general EllipsoidalPotential class - Bovy (UofT)

        """
        EllipsoidalPotential.__init__(
            self,
            amp=amp,
            b=b,
            c=c,
            zvec=zvec,
            pa=pa,
            glorder=glorder,
            ro=ro,
            vo=vo,
            amp_units="mass",
        )
        a = conversion.parse_length(a, ro=self._ro)
        self.a = a
        self._scale = self.a
        # Adjust amp
        self.a4 = self.a**4
        self._amp /= 4.0 * numpy.pi * self.a**3
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            self.normalize(normalize)
        self.hasC = not self._glorder is None
        self.hasC_dxdv = False
        self.hasC_dens = self.hasC  # works if mdens is defined, necessary for hasC
        return None

    def _psi(self, m):
        """\\psi(m) = -\\int_m^\\infty d m^2 \rho(m^2)"""
        return -self.a4 / (m + self.a) ** 2.0

    def _mdens(self, m):
        """Density as a function of m"""
        return self.a4 / m / (m + self.a) ** 3

    def _mdens_deriv(self, m):
        """Derivative of the density as a function of m"""
        return -self.a4 * (self.a + 4.0 * m) / m**2 / (self.a + m) ** 4

    def _mass(self, R, z=None, t=0.0):
        if not z is None:
            raise AttributeError  # Hack to fall back to general
        return (
            4.0
            * numpy.pi
            * self.a4
            / self.a
            / (1.0 + self.a / R) ** 2.0
            / 2.0
            * self._b
            * self._c
        )


class TriaxialJaffePotential(EllipsoidalPotential):
    """Class that implements the Jaffe potential

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)^2\\,(1+m/a)^{2}}

    with

    .. math::

        m^2 = x'^2 + \\frac{y'^2}{b^2}+\\frac{z'^2}{c^2}

    and :math:`(x',y',z')` is a rotated frame wrt :math:`(x,y,z)` specified by parameters ``zvec`` and ``pa`` which specify (a) ``zvec``: the location of the :math:`z'` axis in the :math:`(x,y,z)` frame and (b) ``pa``: the position angle of the :math:`x'` axis wrt the :math:`\\tilde{x}` axis, that is, the :math:`x` axis after rotating to ``zvec``.

    """

    def __init__(
        self,
        amp=1.0,
        a=2.0,
        b=1.0,
        c=1.0,
        zvec=None,
        pa=None,
        normalize=False,
        glorder=50,
        ro=None,
        vo=None,
    ):
        """
        Two-power-law triaxial potential

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass
        a : float or Quantity, optional
            Scale radius.
        b : float, optional
            y-to-x axis ratio of the density
        c : float, optional
            z-to-x axis ratio of the density
        zvec : numpy.ndarray, optional
            If set, a unit vector that corresponds to the z axis
        pa : float or Quantity, optional
            If set, the position angle of the x axis
        glorder : int, optional
            If set, compute the relevant force and potential integrals with Gaussian quadrature of this order
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2010-07-09 - Written - Bovy (UofT)
        - 2018-08-07 - Re-written using the general EllipsoidalPotential class - Bovy (UofT)

        """
        EllipsoidalPotential.__init__(
            self,
            amp=amp,
            b=b,
            c=c,
            zvec=zvec,
            pa=pa,
            glorder=glorder,
            ro=ro,
            vo=vo,
            amp_units="mass",
        )
        a = conversion.parse_length(a, ro=self._ro)
        self.a = a
        self._scale = self.a
        # Adjust amp
        self.a2 = self.a**2
        self._amp /= 4.0 * numpy.pi * self.a2 * self.a
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = not self._glorder is None
        self.hasC_dxdv = False
        self.hasC_dens = self.hasC  # works if mdens is defined, necessary for hasC
        return None

    def _psi(self, m):
        """\\psi(m) = -\\int_m^\\infty d m^2 \rho(m^2)"""
        return (
            2.0
            * self.a2
            * (1.0 / (1.0 + m / self.a) + numpy.log(1.0 / (1.0 + self.a / m)))
        )

    def _mdens(self, m):
        """Density as a function of m"""
        return self.a2 / m**2 / (1.0 + m / self.a) ** 2

    def _mdens_deriv(self, m):
        """Derivative of the density as a function of m"""
        return -2.0 * self.a2**2 * (self.a + 2.0 * m) / m**3 / (self.a + m) ** 3

    def _mass(self, R, z=None, t=0.0):
        if not z is None:
            raise AttributeError  # Hack to fall back to general
        return (
            4.0 * numpy.pi * self.a * self.a2 / (1.0 + self.a / R) * self._b * self._c
        )


class TriaxialNFWPotential(EllipsoidalPotential):
    """Class that implements the triaxial NFW potential

    .. math::

        \\rho(x,y,z) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(m/a)\\,(1+m/a)^{2}}

    with

    .. math::

        m^2 = x'^2 + \\frac{y'^2}{b^2}+\\frac{z'^2}{c^2}

    and :math:`(x',y',z')` is a rotated frame wrt :math:`(x,y,z)` specified by parameters ``zvec`` and ``pa`` which specify (a) ``zvec``: the location of the :math:`z'` axis in the :math:`(x,y,z)` frame and (b) ``pa``: the position angle of the :math:`x'` axis wrt the :math:`\\tilde{x}` axis, that is, the :math:`x` axis after rotating to ``zvec``.

    """

    def __init__(
        self,
        amp=1.0,
        a=2.0,
        b=1.0,
        c=1.0,
        zvec=None,
        pa=None,
        normalize=False,
        conc=None,
        mvir=None,
        glorder=50,
        vo=None,
        ro=None,
        H=70.0,
        Om=0.3,
        overdens=200.0,
        wrtcrit=False,
    ):
        """
        Initialize a triaxial NFW potential

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass
        a : float or Quantity, optional
            Scale radius.
        b : float, optional
            y-to-x axis ratio of the density
        c : float, optional
            z-to-x axis ratio of the density
        zvec : numpy.ndarray, optional
            If set, a unit vector that corresponds to the z axis
        pa : float or Quantity, optional
            If set, the position angle of the x axis
        glorder : int, optional
            If set, compute the relevant force and potential integrals with Gaussian quadrature of this order
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        conc : float, optional
            Concentration.
        mvir : float, optional
            Virial mass in 10^12 Msolar.
        H : float, optional
            Hubble constant in km/s/Mpc.
        Om : float, optional
            Omega matter.
        overdens : float, optional
            Overdensity which defines the virial radius.
        wrtcrit : bool, optional
            If True, the overdensity is wrt the critical density rather than the mean matter density.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Initialize with one of:
            * a and amp or normalize
            * mvir, conc, H, Om, wrtcrit, overdens.
        - 2010-07-09 - Written - Bovy (UofT)
        - 2018-08-07 - Re-written using the general EllipsoidalPotential class - Bovy (UofT)
        """
        EllipsoidalPotential.__init__(
            self,
            amp=amp,
            b=b,
            c=c,
            zvec=zvec,
            pa=pa,
            glorder=glorder,
            ro=ro,
            vo=vo,
            amp_units="mass",
        )
        a = conversion.parse_length(a, ro=self._ro)
        if conc is None:
            self.a = a
        else:
            from ..potential import NFWPotential

            dumb = NFWPotential(
                mvir=mvir,
                conc=conc,
                ro=self._ro,
                vo=self._vo,
                H=H,
                Om=Om,
                wrtcrit=wrtcrit,
                overdens=overdens,
            )
            self.a = dumb.a
            self._amp = dumb._amp
        self._scale = self.a
        self.hasC = not self._glorder is None
        self.hasC_dxdv = False
        self.hasC_dens = self.hasC  # works if mdens is defined, necessary for hasC
        # Adjust amp
        self.a3 = self.a**3
        self._amp /= 4.0 * numpy.pi * self.a3
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            self.normalize(normalize)
        return None

    def _psi(self, m):
        """\\psi(m) = -\\int_m^\\infty d m^2 \rho(m^2)"""
        return -2.0 * self.a3 / (self.a + m)

    def _mdens(self, m):
        """Density as a function of m"""
        return self.a / m / (1.0 + m / self.a) ** 2

    def _mdens_deriv(self, m):
        """Derivative of the density as a function of m"""
        return -self.a3 * (self.a + 3.0 * m) / m**2 / (self.a + m) ** 3

    def _mass(self, R, z=None, t=0.0):
        if not z is None:
            raise AttributeError  # Hack to fall back to general
        return (
            4.0
            * numpy.pi
            * self.a3
            * self._b
            * self._c
            * (numpy.log(1 + R / self.a) - R / self.a / (1.0 + R / self.a))
        )
