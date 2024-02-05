###############################################################################
#   LogarithmicHaloPotential.py: class that implements the logarithmic
#                            potential Phi(r) = vc**2 ln(r)
###############################################################################
import warnings

import numpy

from ..util import conversion, galpyWarning
from .Potential import Potential, kms_to_kpcGyrDecorator

_CORE = 10**-8


class LogarithmicHaloPotential(Potential):
    """Class that implements the logarithmic potential

    .. math::

        \\Phi(R,z) = \\frac{\\mathrm{amp}}{2}\\,\\ln\\left[R^2+\\left(\\frac{z}{q}\\right)^2+\\mathrm{core}^2\\right]

    Alternatively, the potential can be made triaxial by adding a parameter :math:`b`

    .. math::

        \\Phi(x,y,z) = \\frac{\\mathrm{amp}}{2}\\,\\ln\\left[x^2+\\left(\\frac{y}{b}\\right)^2+\\left(\\frac{z}{q}\\right)^2+\\mathrm{core}^2\\right]

    With these definitions, :math:`\\sqrt{\\mathrm{amp}}` is the circular velocity at :math:`r \\gg \\mathrm{core}` at :math:`(y,z) = (0,0)`.

    """

    def __init__(
        self, amp=1.0, core=_CORE, q=1.0, b=None, normalize=False, ro=None, vo=None
    ):
        """
        Initialize a logarithmic potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential; can be a Quantity with units of velocity-squared.
        core : float or Quantity, optional
            Core radius at which the logarithm is cut.
        q : float
            Potential flattening (z/q)**2.
        b : float, optional
            Shape parameter in y-direction (y --> y/b; see definition).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2010-04-02 - Started - Bovy (NYU)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="velocity2")
        core = conversion.parse_length(core, ro=self._ro)
        self.hasC = True
        self.hasC_dxdv = True
        self.hasC_dens = True
        self._core2 = core**2.0
        self._q = q
        self._b = b
        if not self._b is None:
            self.isNonAxi = True
            self._1m1overb2 = 1.0 - 1.0 / self._b**2.0
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self._nemo_accname = "LogPot"
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            return (
                1.0
                / 2.0
                * numpy.log(
                    R**2.0 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
                    + (z / self._q) ** 2.0
                    + self._core2
                )
            )
        else:
            return 1.0 / 2.0 * numpy.log(R**2.0 + (z / self._q) ** 2.0 + self._core2)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            Rt2 = R**2.0 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
            return -Rt2 / R / (Rt2 + (z / self._q) ** 2.0 + self._core2)
        else:
            return -R / (R**2.0 + (z / self._q) ** 2.0 + self._core2)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            Rt2 = R**2.0 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
            return -z / self._q**2.0 / (Rt2 + (z / self._q) ** 2.0 + self._core2)
        else:
            return -z / self._q**2.0 / (R**2.0 + (z / self._q) ** 2.0 + self._core2)

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            Rt2 = R**2.0 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
            return (
                R**2.0
                / (Rt2 + (z / self._q) ** 2.0 + self._core2)
                * numpy.sin(2.0 * phi)
                * self._1m1overb2
                / 2.0
            )
        else:
            return 0

    def _dens(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            R2 = R**2.0
            Rt2 = R2 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
            denom = 1.0 / (Rt2 + (z / self._q) ** 2.0 + self._core2)
            denom2 = denom**2.0
            return (
                1.0
                / 4.0
                / numpy.pi
                * (
                    2.0 * Rt2 / R2 * (denom - Rt2 * denom2)
                    + denom / self._q**2.0
                    - 2.0 * z**2.0 * denom2 / self._q**4.0
                    - self._1m1overb2
                    * (
                        2.0
                        * R2
                        * numpy.sin(2.0 * phi) ** 2.0
                        / 4.0
                        * self._1m1overb2
                        * denom2
                        + denom * numpy.cos(2.0 * phi)
                    )
                )
            )
        else:
            return (
                1.0
                / 4.0
                / numpy.pi
                / self._q**2.0
                * (
                    (2.0 * self._q**2.0 + 1.0) * self._core2
                    + R**2.0
                    + (2.0 - self._q**-2.0) * z**2.0
                )
                / (R**2.0 + (z / self._q) ** 2.0 + self._core2) ** 2.0
            )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            Rt2 = R**2.0 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
            denom = 1.0 / (Rt2 + (z / self._q) ** 2.0 + self._core2)
            return (denom - 2.0 * Rt2 * denom**2.0) * Rt2 / R**2.0
        else:
            denom = 1.0 / (R**2.0 + (z / self._q) ** 2.0 + self._core2)
            return denom - 2.0 * R**2.0 * denom**2.0

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            Rt2 = R**2.0 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
            denom = 1.0 / (Rt2 + (z / self._q) ** 2.0 + self._core2)
            return denom / self._q**2.0 - 2.0 * z**2.0 * denom**2.0 / self._q**4.0
        else:
            denom = 1.0 / (R**2.0 + (z / self._q) ** 2.0 + self._core2)
            return denom / self._q**2.0 - 2.0 * z**2.0 * denom**2.0 / self._q**4.0

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            Rt2 = R**2.0 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
            return (
                -2.0
                * Rt2
                / R
                * z
                / self._q**2.0
                / (Rt2 + (z / self._q) ** 2.0 + self._core2) ** 2.0
            )
        else:
            return (
                -2.0
                * R
                * z
                / self._q**2.0
                / (R**2.0 + (z / self._q) ** 2.0 + self._core2) ** 2.0
            )

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            Rt2 = R**2.0 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
            denom = 1.0 / (Rt2 + (z / self._q) ** 2.0 + self._core2)
            return -self._1m1overb2 * (
                R**4.0
                * numpy.sin(2.0 * phi) ** 2.0
                / 2.0
                * self._1m1overb2
                * denom**2.0
                + R**2.0 * denom * numpy.cos(2.0 * phi)
            )
        else:
            return 0.0

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            Rt2 = R**2.0 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
            denom = 1.0 / (Rt2 + (z / self._q) ** 2.0 + self._core2)
            return (
                -(denom - Rt2 * denom**2.0) * R * numpy.sin(2.0 * phi) * self._1m1overb2
            )
        else:
            return 0.0

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        if self.isNonAxi:
            Rt2 = R**2.0 * (1.0 - self._1m1overb2 * numpy.sin(phi) ** 2.0)
            denom = 1.0 / (Rt2 + (z / self._q) ** 2.0 + self._core2)
            return (
                2
                * R**2
                * z
                * numpy.sin(phi)
                * numpy.cos(phi)
                * self._1m1overb2
                * denom**2
                / self._q**2
            )
        else:
            return 0.0

    @kms_to_kpcGyrDecorator
    def _nemo_accpars(self, vo, ro):
        warnings.warn(
            "NEMO's LogPot does not allow flattening in z (for some reason); therefore, flip y and z in NEMO wrt galpy; also does not allow the triaxial b parameter",
            galpyWarning,
        )
        ampl = self._amp * vo**2.0
        return "0,{},{},1.0,{}".format(
            ampl,
            self._core2
            * ro**2.0
            * self._q ** (2.0 / 3.0),  # somewhat weird gyrfalcon implementation
            self._q,
        )
