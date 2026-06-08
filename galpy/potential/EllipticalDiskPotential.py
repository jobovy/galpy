###############################################################################
#   EllipticalDiskPotential: Kuijken & Tremaine (1994)'s elliptical disk
#   potential
###############################################################################
import numpy

from ..backend import get_namespace
from ..util import conversion
from .planarPotential import planarPotential

_degtorad = numpy.pi / 180.0


class EllipticalDiskPotential(planarPotential):
    """Class that implements the Elliptical disk potential of Kuijken & Tremaine (1994)

    .. math::

        \\Phi(R,\\phi) = \\mathrm{amp}\\,\\phi_0\\,\\left(\\frac{R}{R_1}\\right)^p\\,\\cos\\left(2\\,(\\phi-\\phi_b)\\right)

    This potential can be grown between  :math:`t_{\\mathrm{form}}` and  :math:`t_{\\mathrm{form}}+T_{\\mathrm{steady}}` in a similar way as DehnenBarPotential, but times are given directly in galpy time units

    """

    def __init__(
        self,
        amp=1.0,
        phib=25.0 * _degtorad,
        p=1.0,
        twophio=0.01,
        r1=1.0,
        tform=None,
        tsteady=None,
        cp=None,
        sp=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize an Elliptical disk potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1.), see twophio below.
        tform : float or Quantity, optional
            Start of growth (to smoothly grow this potential).
        tsteady : float or Quantity, optional
            Time delay at which the perturbation is fully grown (default: 2.).
        p : float, optional
            Power-law index of the phi(R) = (R/Ro)^p part.
        r1 : float or Quantity, optional
            Normalization radius for the amplitude.
        phib : float or Quantity, optional
            Angle (in rad; default=25 degree).
        twophio : float or Quantity, optional
            Potential perturbation (in terms of 2phio/vo^2 if vo=1 at Ro=1).
        cp : float or Quantity, optional
            Twophio * cos(2phib).
        sp : float or Quantity, optional
            Twophio * sin(2phib).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Specify either (phib,twophio) or (cp,sp).
        - 2011-10-19 - Started - Bovy (IAS)

        """
        planarPotential.__init__(self, amp=amp, ro=ro, vo=vo)
        phib = conversion.parse_angle(phib)
        r1 = conversion.parse_length(r1, ro=self._ro)
        tform = conversion.parse_time(tform, ro=self._ro, vo=self._vo)
        tsteady = conversion.parse_time(tsteady, ro=self._ro, vo=self._vo)
        twophio = conversion.parse_energy(twophio, vo=self._vo)
        cp = conversion.parse_energy(cp, vo=self._vo)
        sp = conversion.parse_energy(sp, vo=self._vo)
        # Back to old definition
        self._amp /= r1**p
        self.hasC = True
        self.hasC_dxdv = True
        if cp is None or sp is None:
            self._phib = phib
            self._twophio = twophio
        else:
            self._twophio = numpy.sqrt(cp * cp + sp * sp)
            self._phib = numpy.arctan2(sp, cp) / 2.0
        self._p = p
        if not tform is None:
            self._tform = tform
        else:
            self._tform = None
        if not tsteady is None:
            self._tsteady = self._tform + tsteady
        else:
            if self._tform is None:
                self._tsteady = None
            else:
                self._tsteady = self._tform + 2.0

    def _smooth(self, t):
        # Growth factor (0 before tform, smoothly to 1 at tsteady). The namespace
        # follows t itself: a concrete (Python/numpy) t -> numpy.where (a plain
        # coefficient, byte-identical to the original branches, that broadcasts
        # into any backend's spatial arrays); a traced t (the in-backend
        # diffrax/torchdiffeq integrator, or autodiff wrt time) -> that backend's
        # where, so it is differentiable. Branch-free so a tracer works.
        if self._tform is None:
            return 1.0
        xp = get_namespace(t)
        deltat = t - self._tform
        xi = 2.0 * deltat / (self._tsteady - self._tform) - 1.0
        growth = 3.0 / 16.0 * xi**5.0 - 5.0 / 8 * xi**3.0 + 15.0 / 16.0 * xi + 0.5
        return xp.where(t < self._tform, 0.0, xp.where(t < self._tsteady, growth, 1.0))

    def _evaluate(self, R, phi=0.0, t=0.0):
        xp = get_namespace(R, phi, t)
        smooth = self._smooth(t)
        return (
            smooth * self._twophio / 2.0 * R**self._p * xp.cos(2.0 * (phi - self._phib))
        )

    def _Rforce(self, R, phi=0.0, t=0.0):
        xp = get_namespace(R, phi, t)
        smooth = self._smooth(t)
        return (
            -smooth
            * self._p
            * self._twophio
            / 2.0
            * R ** (self._p - 1.0)
            * xp.cos(2.0 * (phi - self._phib))
        )

    def _phitorque(self, R, phi=0.0, t=0.0):
        xp = get_namespace(R, phi, t)
        smooth = self._smooth(t)
        return smooth * self._twophio * R**self._p * xp.sin(2.0 * (phi - self._phib))

    def _R2deriv(self, R, phi=0.0, t=0.0):
        xp = get_namespace(R, phi, t)
        smooth = self._smooth(t)
        return (
            smooth
            * self._p
            * (self._p - 1.0)
            / 2.0
            * self._twophio
            * R ** (self._p - 2.0)
            * xp.cos(2.0 * (phi - self._phib))
        )

    def _phi2deriv(self, R, phi=0.0, t=0.0):
        xp = get_namespace(R, phi, t)
        smooth = self._smooth(t)
        return (
            -2.0
            * smooth
            * self._twophio
            * R**self._p
            * xp.cos(2.0 * (phi - self._phib))
        )

    def _Rphideriv(self, R, phi=0.0, t=0.0):
        xp = get_namespace(R, phi, t)
        smooth = self._smooth(t)
        return (
            -smooth
            * self._p
            * self._twophio
            * R ** (self._p - 1.0)
            * xp.sin(2.0 * (phi - self._phib))
        )

    def tform(self):  # pragma: no cover
        """
        Return formation time of the perturbation.

        Returns
        -------
        float
            Formation time of the perturbation in normalized units.

        Notes
        -----
        - 2011-10-19 - Written - Bovy (IAS)
        """
        return self._tform
