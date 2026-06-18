###############################################################################
#   ExpTruncNFWPotential.py: NFW potential with an exponential truncation
###############################################################################
import numpy
from scipy.special import exp1

from ..util import conversion
from .SphericalPotential import SphericalPotential


class ExpTruncNFWPotential(SphericalPotential):
    r"""Class that implements the exponentially-truncated NFW potential

    .. math::

        \rho(r) = \frac{\mathrm{amp}}{4\,\pi\,a^3}\,\frac{e^{-r/r_c}}{(r/a)\,(1+r/a)^{2}}

    i.e., the :ref:`NFW <nfw>` profile multiplied by an exponential cutoff with
    truncation radius :math:`r_c`. All methods are closed-form: the enclosed
    mass and the outer-potential integral are expressible through the
    exponential integral :math:`E_1` (:func:`scipy.special.exp1`), with a
    small-:math:`r` Taylor expansion used to avoid catastrophic cancellation
    when :math:`r \ll a, r_c`.

    """

    def __init__(
        self, amp=1.0, a=1.0, rc=2.0, mass=None, normalize=False, ro=None, vo=None
    ):
        """
        Initialize an exponentially-truncated NFW potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass. This is the NFW mass-scale :math:`4\\pi\\,\\rho_s\\,a^3`, matching the ``amp`` convention of :ref:`NFWPotential <nfw>` (in the :math:`r_c \\to \\infty` limit the two potentials coincide for equal ``amp`` and ``a``). Ignored if ``mass`` is set.
        a : float or Quantity, optional
            Scale radius (can be Quantity).
        rc : float or Quantity, optional
            Exponential truncation radius (can be Quantity).
        mass : float or Quantity, optional
            Total mass of the (finite-mass) profile; if set, ``amp`` is determined from it as ``amp = mass / [exp(a/rc)(1+a/rc)E_1(a/rc) - 1]`` and the ``amp`` argument is ignored.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Initialize with one of: ``a`` and (``amp`` or ``normalize``); or ``a`` and ``mass``.
        - The closed-form total mass becomes unevaluable in floating point for an
          extremely sharp truncation, ``a / rc`` :math:`\\gtrsim 700` (i.e., ``rc``
          several hundred times smaller than ``a``); this is far outside any
          physically sensible choice of ``rc`` (which should be comparable to or
          larger than ``a``), so no explicit check is made for it.
        - 2026-06-16 - Written - Pfaffman + Claude Code

        """
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        rc = conversion.parse_length(rc, ro=self._ro)
        self.a = a
        self.rc = rc
        self._scale = self.a
        # Precompute quantities involving the truncation that are reused by the
        # closed-form mass and potential integrals.
        self._alpha = a / rc
        self._exp_alpha = numpy.exp(self._alpha)
        self._E1_alpha = exp1(self._alpha)
        # Dimensionless total mass M_tot/amp = F(infinity): as r->inf the two E1
        # terms of the closed-form F(r) leave exp(alpha)(1+alpha)E1(alpha) - 1.
        self._Ftot = self._exp_alpha * (1.0 + self._alpha) * self._E1_alpha - 1.0
        if mass is not None:
            newmass = conversion.parse_mass(mass, ro=self._ro, vo=self._vo)
            if newmass != mass:  # a Quantity was passed; report physical units
                self.turn_physical_on(ro=self._ro, vo=self._vo)
            self._amp = newmass / self._Ftot
        # Threshold below which the closed-form M(<r) suffers cancellation; use
        # the series expansion instead. r/min(a, rc) < eps gives roughly eps^2
        # relative truncation error in F(r).
        self._small_r_thresh = 1e-3 * min(a, rc)
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = False
        self.hasC_dxdv = False
        return None

    def _F(self, r):
        # F(r) = M(<r) / amp = int_0^r s e^{-s/rc} / (a+s)^2 ds, the
        # dimensionless enclosed-mass scale (so that _rforce = -F(r)/r^2 in
        # amp-units). For r << a, rc the two E1 terms cancel; use a Taylor
        # series in r there instead.
        r = numpy.asarray(r, dtype=float)
        if r.ndim == 0:
            return self._F_scalar(float(r))
        small = r < self._small_r_thresh
        out = numpy.empty_like(r)
        out[~small] = self._F_closed(r[~small])
        out[small] = self._F_series(r[small])
        return out

    def _F_scalar(self, r):
        if r < self._small_r_thresh:
            return self._F_series(r)
        return self._F_closed(r)

    def _F_closed(self, r):
        # F(r) = exp(alpha)(1+alpha)[E1(alpha) - E1(beta)] - 1
        #        + a exp(-r/rc)/(a+r),
        # with alpha = a/rc and beta = (a+r)/rc.
        a, rc = self.a, self.rc
        beta = (a + r) / rc
        return (
            self._exp_alpha * (1.0 + self._alpha) * (self._E1_alpha - exp1(beta))
            - 1.0
            + a * numpy.exp(-r / rc) / (a + r)
        )

    def _F_series(self, r):
        # Taylor expansion of F(r) about r=0 (through O(r^5)):
        # F(r) = (1/a^2) [ r^2/2 - (r^3/3)(1/rc + 2/a)
        #                + (r^4/4)(1/(2 rc^2) + 2/(rc a) + 3/a^2)
        #                - (r^5/5)(1/(6 rc^3) + 1/(rc^2 a)
        #                         + 3/(rc a^2) + 4/a^3) + ... ]
        a, rc = self.a, self.rc
        c2 = 0.5
        c3 = -(1.0 / rc + 2.0 / a) / 3.0
        c4 = (0.5 / rc / rc + 2.0 / (rc * a) + 3.0 / (a * a)) / 4.0
        c5 = (
            -(
                1.0 / (6.0 * rc**3)
                + 1.0 / (rc * rc * a)
                + 3.0 / (rc * a * a)
                + 4.0 / a**3
            )
            / 5.0
        )
        return (r * r / (a * a)) * (c2 + r * (c3 + r * (c4 + r * c5)))

    def _G(self, r):
        # G(r) := 4 pi int_r^inf rho(s) s ds / amp
        #       = exp(-r/rc)/(a+r) - exp(alpha) E1(beta) / rc,
        # the outer-shell contribution to the potential.
        a, rc = self.a, self.rc
        beta = (a + r) / rc
        return numpy.exp(-r / rc) / (a + r) - self._exp_alpha * exp1(beta) / rc

    def _rdens(self, r, t=0.0):
        # rho(r) / amp; the 1/(4 pi a^3) factor is carried here so that the
        # public dens(r) = rho_s exp(-r/rc) / [(r/a)(1+r/a)^2], matching the
        # NFW amplitude convention.
        a = self.a
        return numpy.exp(-r / self.rc) / (
            4.0 * numpy.pi * a * a * r * (1.0 + r / a) ** 2
        )

    def _revaluate(self, r, t=0.0):
        # Phi(r)/amp = -[F(r)/r + G(r)]. F(r) ~ r^2/(2 a^2) near the origin, so
        # F/r has a finite (zero) limit there that we substitute by hand to
        # avoid a 0/0 NaN (e.g. eddingtondf seeds its rphi spline at r=0).
        r = numpy.asarray(r, dtype=float)
        if r.ndim == 0:
            if r == 0.0:
                return -self._G(0.0)
            return -(self._F(r) / r + self._G(r))
        out = -self._G(r)
        nz = r != 0.0
        out[nz] -= self._F(r[nz]) / r[nz]
        return out

    def _rforce(self, r, t=0.0):
        # -F(r)/r^2, with the finite r->0 limit -1/(2 a^2) substituted by hand.
        r = numpy.asarray(r, dtype=float)
        zero_limit = -0.5 / (self.a * self.a)
        if r.ndim == 0:
            if r == 0.0:
                return zero_limit
            return -self._F(r) / (r * r)
        out = numpy.full_like(r, zero_limit)
        nz = r != 0.0
        out[nz] = -self._F(r[nz]) / (r[nz] * r[nz])
        return out

    def _r2deriv(self, r, t=0.0):
        return 4.0 * numpy.pi * self._rdens(r) - 2.0 * self._F(r) / r**3

    def _mass(self, R, z=None, t=0.0):
        # Closed-form enclosed mass M(<R)/amp = F(R). Overriding the generic
        # -R^2*rforce(R) lets mass(numpy.inf) return the finite total mass
        # amp*F(inf) instead of inf^2*0 = NaN (F_closed evaluates finitely at
        # infinity: exp1(inf)=0 and exp(-inf)=0).
        if z is not None:
            raise AttributeError  # use general (spheroidal) implementation
        return self._F(numpy.asarray(R, dtype=float))

    def _ddensdr(self, r, t=0.0):
        # galpy calls _ddensdr/_d2densdr2 with amp already applied, so bake in
        # self._amp here (matching the TwoPowerSphericalPotential convention).
        rho_phys = self._amp * self._rdens(r)
        g = 1.0 / r + 2.0 / (self.a + r) + 1.0 / self.rc
        return -rho_phys * g

    def _d2densdr2(self, r, t=0.0):
        rho_phys = self._amp * self._rdens(r)
        g = 1.0 / r + 2.0 / (self.a + r) + 1.0 / self.rc
        gprime = -1.0 / (r * r) - 2.0 / (self.a + r) ** 2
        return rho_phys * (g * g - gprime)
