###############################################################################
#   ExpTruncNFWPotential.py: NFW potential with an exponential truncation
###############################################################################
import warnings

import numpy
from scipy.special import exp1 as _scipy_exp1

from ..backend import as_numpy, coerce_coords, get_namespace, is_backend_array
from ..backend._namespaces import namespace_from_arrays
from ..backend.special import exp1
from ..util import conversion, galpyWarning
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
    when :math:`r \ll a, r_c`. Has a C implementation, enabling fast orbit
    integration and the full 3D variational equations (``Orbit.integrate_dxdv``).

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
        # Under a forced backend these arrive as plain floats, so alpha below
        # would be a float and construction would run on scipy -- the potential
        # would never be built ON the forced backend. coerce_coords is a strict
        # pass-through when xp is numpy, so the numpy path stays byte-identical.
        a, rc = coerce_coords(get_namespace(a, rc), a, rc)
        self.a = a
        self.rc = rc
        # _scale is grid-construction bookkeeping, not physics: sphericaldf
        # consumes it as a plain scalar length (numpy.log10(rmax/_scale),
        # r_a_values * _scale, ...), so a coerced backend value there raises
        # "unsupported operand type(s) for *: 'numpy.ndarray' and 'Tensor'".
        # Keep it numpy; the physical scale stays on the backend.
        self._scale = float(as_numpy(self.a)) if is_backend_array(self.a) else self.a
        # Precompute quantities involving the truncation that are reused by the
        # closed-form mass and potential integrals.
        self._alpha = a / rc
        # data-first: a backend a/rc keeps the potential differentiable w.r.t.
        # the truncation parameters. Under a forced backend a/rc were coerced
        # above, so this takes the backend branch there too; an unforced plain
        # float still stays on scipy and byte-identical.
        if is_backend_array(self._alpha):
            xp = get_namespace(self._alpha)
            self._exp_alpha = xp.exp(self._alpha)
            self._E1_alpha = exp1(self._alpha)
        else:
            self._exp_alpha = numpy.exp(self._alpha)
            self._E1_alpha = _scipy_exp1(self._alpha)
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
        self.hasC = True
        self._backend_compatible = True
        self.hasC_dxdv = True
        self.hasC_dxdv3d = True
        self.hasC_dens = True
        return None

    @classmethod
    def from_nfw(cls, nfw, rc=None, mass=None):
        """
        Initialize an ExpTruncNFWPotential by truncating an existing NFWPotential.

        The amplitude ``amp`` and scale radius ``a`` are inherited from ``nfw``,
        so the inner profile is unchanged (the truncated density is just the NFW
        density times :math:`e^{-r/r_c}`). The truncation is set by **one** of:

        - ``rc``: the truncation radius directly (e.g. the virial radius
          ``nfw.rvir()``); the total mass then follows; or
        - ``mass``: the desired (finite) total mass, from which ``rc`` is solved
          as the unique root of ``nfw.amp * [exp(a/rc)(1+a/rc)E_1(a/rc) - 1] =
          mass`` (``amp`` is still inherited, so this just chooses where to
          truncate). ``rc`` and ``mass`` are interchangeable here and exactly one
          must be given.

        Parameters
        ----------
        nfw : NFWPotential
            The NFW potential to truncate; its ``amp`` and ``a`` (and its unit
            system) are inherited.
        rc : float or Quantity, optional
            Exponential truncation radius. Provide this or ``mass``, not both.
        mass : float or Quantity, optional
            Desired total mass; ``rc`` is computed from it. Provide this or
            ``rc``, not both.

        Returns
        -------
        ExpTruncNFWPotential
            A truncated NFW with the same inner profile as ``nfw``.

        Raises
        ------
        TypeError
            If ``nfw`` is not an ``NFWPotential`` instance.
        ValueError
            If neither or both of ``rc`` and ``mass`` are given; or if ``mass``
            is given but is so small that it would require a truncation sharper
            than ``rc < a/690``, where the closed-form total mass overflows in
            floating point (an unphysically sharp truncation). Note there is no
            upper limit on ``mass``: the un-truncated NFW has infinite mass, so
            any finite mass is reachable by truncating far enough out.

        Warns
        -----
        galpyWarning
            If ``mass`` is given and the solved truncation radius is either much
            smaller than the NFW scale radius (``rc < a`` -- a very sharp
            truncation cutting into the cusp) or well beyond the NFW virial
            radius (``rc > 2 rvir`` -- a weak truncation that retains much of the
            formally-divergent NFW outskirts).

        Notes
        -----
        - 2026-06-24 - Written - Pfaffman + Claude Code

        """
        from .TwoPowerSphericalPotential import NFWPotential

        if not isinstance(nfw, NFWPotential):
            raise TypeError(
                "ExpTruncNFWPotential.from_nfw requires an NFWPotential instance"
            )
        if (rc is None) == (mass is None):
            raise ValueError(
                "ExpTruncNFWPotential.from_nfw requires exactly one of rc or mass"
            )
        a = nfw.a
        if mass is not None:
            # Keep amp = nfw.amp and solve amp * F(a/rc) = mass for rc, where
            # F(alpha) = exp(alpha)(1+alpha)E_1(alpha) - 1 decreases monotonically
            # from +inf (alpha->0) to 0 (alpha->inf), so the root is unique.
            from ..backend.optimize import brentq

            target_F = conversion.parse_mass(mass, ro=nfw._ro, vo=nfw._vo) / nfw._amp
            # Namespace-agnostic residual, with target_F passed through ``args``
            # rather than closed over: galpy's brentq follows the DATA, and only
            # an argument it can see selects the differentiable backend path.
            # xp must follow the DATA, not the ambient (possibly FORCED) default:
            # under `use("torch", force=True)` with a plain-float mass,
            # get_namespace would say torch while brentq -- which dispatches on
            # the data -- correctly goes to scipy and hands the residual Python
            # floats, giving `torch.exp(float)`. Deriving both from the same
            # predicate makes xp and the brentq dispatch agree by construction.
            numpy_path = not is_backend_array(target_F)
            xp = numpy if numpy_path else get_namespace(target_F)
            # exp1 ALSO follows the ambient namespace, so on the numpy path take
            # scipy's directly: the router would return a torch Tensor for a
            # Python-float argument under `use("torch", force=True)`, and scipy's
            # brentq would then be handed a Tensor residual.
            _e1 = _scipy_exp1 if numpy_path else exp1
            Froot = lambda al, tF: xp.exp(al) * (1.0 + al) * _e1(al) - 1.0 - tF
            # numpy stays byte-identical: xp is numpy, exp1 routes to
            # scipy.special.exp1, and brentq routes to scipy.optimize.brentq.
            # On a backend target_F the value is a TRACER under jax.grad, so the
            # bracket pre-check and the advisory warnings below -- both ordinary
            # Python control flow on the solved value -- are numpy-only. The
            # trade is deliberate: gradients in exchange for the domain check
            # and the warnings, which are user guidance rather than correctness.
            # F(alpha) decreases monotonically from +inf (alpha->0, rc->inf, the
            # un-truncated infinite-mass NFW) to 0 (alpha->inf). Any finite mass
            # is therefore reachable, with a larger mass simply giving a larger
            # rc -- there is no upper bound. The only hard limit is at the sharp
            # end: F(690) ~ 2e-6 is the smallest mass we can evaluate before
            # exp(alpha) overflows (rc < a/690).
            a_low, a_high = 1e-150, 690.0  # F(1e-150) ~ 344; exp(690) still finite
            if numpy_path and Froot(a_high, target_F) > 0.0:
                raise ValueError(
                    "ExpTruncNFWPotential.from_nfw: the requested mass is too "
                    "small to evaluate -- it implies a truncation sharper than "
                    "rc = a/690, where the closed-form total mass overflows in "
                    "floating point"
                )
            alpha = brentq(Froot, a_low, a_high, args=(target_F,))
            rc = a / alpha
            if numpy_path and rc < a:
                warnings.warn(
                    "ExpTruncNFWPotential.from_nfw: the requested total mass "
                    f"implies a truncation radius rc={rc:g} smaller than the NFW "
                    f"scale radius a={a:g} (a/rc={alpha:g}); this is a very sharp "
                    "truncation that cuts into the NFW cusp -- check that the "
                    "requested mass is intended.",
                    galpyWarning,
                )
            else:
                # Warn if the truncation lands well beyond the virial radius,
                # i.e. a weak truncation that keeps much of the (formally
                # divergent) NFW outskirts. rvir needs the overdensity definition
                # and can fail (e.g. odd unit setups), so guard it.
                try:
                    rvir = nfw.rvir(use_physical=False) if numpy_path else None
                except Exception:  # pragma: no cover
                    rvir = None
                if rvir is not None and rc > 2.0 * rvir:
                    warnings.warn(
                        "ExpTruncNFWPotential.from_nfw: the requested total mass "
                        f"implies a truncation radius rc={rc:g} more than twice "
                        f"the NFW virial radius rvir={rvir:g}; this is a weak "
                        "truncation that retains much of the (formally divergent) "
                        "NFW outskirts -- check that the requested mass is "
                        "intended.",
                        galpyWarning,
                    )
        # Inherit amp, a, and the unit system from the NFW so the inner profile
        # (and the meaning of the internal amp/a) carries over directly; rc is
        # parsed in the NFW's units by the constructor.
        out = cls(amp=nfw._amp, a=a, rc=rc, ro=nfw._ro, vo=nfw._vo)
        # Faithfully reproduce the NFW's physical-output (ro/vo) state.
        out._roSet = nfw._roSet
        out._voSet = nfw._voSet
        return out

    def _F(self, r):
        # F(r) = M(<r) / amp = int_0^r s e^{-s/rc} / (a+s)^2 ds, the
        # dimensionless enclosed-mass scale (so that _rforce = -F(r)/r^2 in
        # amp-units). For r << a, rc the two E1 terms cancel; use a Taylor
        # series in r there instead. Both branches are finite and smooth on the
        # whole domain (the split is for precision only), so the masked-out side
        # cannot NaN-poison AD -- a plain xp.where suffices.
        xp = get_namespace(r)
        r = xp.asarray(r) * 1.0
        small = r < self._small_r_thresh
        return xp.where(small, self._F_series(r), self._F_closed(r))

    def _F_closed(self, r):
        # F(r) = exp(alpha)(1+alpha)[E1(alpha) - E1(beta)] - 1
        #        + a exp(-r/rc)/(a+r),
        # with alpha = a/rc and beta = (a+r)/rc.
        xp = get_namespace(r)
        r = xp.asarray(r) * 1.0  # so beta is a backend array (router/dtype match)
        a, rc = self.a, self.rc
        beta = (a + r) / rc
        return (
            self._exp_alpha * (1.0 + self._alpha) * (self._E1_alpha - exp1(beta))
            - 1.0
            + a * xp.exp(-r / rc) / (a + r)
        )

    def _F_series(self, r):
        # Taylor expansion of F(r) about r=0 (through O(r^5)):
        # F(r) = (1/a^2) [ r^2/2 - (r^3/3)(1/rc + 2/a)
        #                + (r^4/4)(1/(2 rc^2) + 2/(rc a) + 3/a^2)
        #                - (r^5/5)(1/(6 rc^3) + 1/(rc^2 a)
        #                         + 3/(rc a^2) + 4/a^3) + ... ]
        xp = get_namespace(r)
        r = xp.asarray(r) * 1.0  # match _F_closed's namespace (direct-call parity)
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
        xp = get_namespace(r)
        r = xp.asarray(r) * 1.0  # so beta is a backend array (router/dtype match)
        a, rc = self.a, self.rc
        beta = (a + r) / rc
        return xp.exp(-r / rc) / (a + r) - self._exp_alpha * exp1(beta) / rc

    def _rdens(self, r, t=0.0):
        # rho(r) / amp; the 1/(4 pi a^3) factor is carried here so that the
        # public dens(r) = rho_s exp(-r/rc) / [(r/a)(1+r/a)^2], matching the
        # NFW amplitude convention. data-first (dispatch on r's own namespace):
        # _ddensdr feeds the spherical DF machinery, which may pass a tracer
        # under a forced other backend.
        xp = namespace_from_arrays((r,)) or numpy
        r = xp.asarray(r) * 1.0  # so xp.exp gets a backend array (scalar inputs)
        a = self.a
        return xp.exp(-r / self.rc) / (4.0 * numpy.pi * a * a * r * (1.0 + r / a) ** 2)

    def _revaluate(self, r, t=0.0):
        # Phi(r)/amp = -[F(r)/r + G(r)]. F(r) ~ r^2/(2 a^2) near the origin, so
        # F/r has a finite (zero) limit there that we substitute by hand to
        # avoid a 0/0 NaN (e.g. eddingtondf seeds its rphi spline at r=0).
        xp = get_namespace(r)
        r = xp.asarray(r) * 1.0
        at0 = r == 0.0
        safe = xp.where(at0, xp.ones_like(r), r)  # avoid 0/0 at the origin
        FoverR = xp.where(at0, xp.zeros_like(r), self._F(safe) / safe)
        return -(FoverR + self._G(r))

    def _rforce(self, r, t=0.0):
        # -F(r)/r^2, with the finite r->0 limit -1/(2 a^2) substituted by hand.
        xp = get_namespace(r)
        r = xp.asarray(r) * 1.0
        zero_limit = -0.5 / (self.a * self.a)
        at0 = r == 0.0
        safe = xp.where(at0, xp.ones_like(r), r)  # avoid 0/0 at the origin
        force = -self._F(safe) / (safe * safe)
        # zero_limit * ones_like (not full_like): keeps it an array-broadcast of a
        # possibly-Tensor limit when a is a differentiable backend parameter
        # (torch.full_like rejects a Tensor fill); byte-identical for numpy.
        return xp.where(at0, zero_limit * xp.ones_like(r), force)

    def _r2deriv(self, r, t=0.0):
        return 4.0 * numpy.pi * self._rdens(r) - 2.0 * self._F(r) / r**3

    def _mass(self, R, z=None, t=0.0):
        # Closed-form enclosed mass M(<R)/amp = F(R). Overriding the generic
        # -R^2*rforce(R) lets mass(numpy.inf) return the finite total mass
        # amp*F(inf) instead of inf^2*0 = NaN (F_closed evaluates finitely at
        # infinity: exp1(inf)=0 and exp(-inf)=0).
        if z is not None:
            raise AttributeError  # use general (spheroidal) implementation
        xp = get_namespace(R)
        return self._F(xp.asarray(R) * 1.0)

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

    def _ddenstwobetadr(self, r, beta=0):
        """
        Evaluate the radial density derivative x r^(2beta) for this potential.

        Parameters
        ----------
        r : float
            Spherical radius.
        beta : float, optional
            Power of r in the density derivative. Default is 0.

        Returns
        -------
        float
            The derivative of rho x r^(2beta) with respect to r.

        Notes
        -----
        - 2026-06-25 - Written - Pfaffman + Claude Code

        """
        # Consumed (via JAX grad) by the constantbetadf machinery, which passes
        # jax tracers even under a forced-torch run, so dispatch on r's OWN
        # namespace (data-first) rather than the forced default.
        # d/dr[rho r^(2beta)] = rho r^(2beta) [(2beta-1)/r - 2/(a+r) - 1/rc];
        # reduces to _ddensdr at beta=0.
        xp = namespace_from_arrays((r,)) or numpy
        a, rc = self.a, self.rc
        rho = (
            self._amp
            * xp.exp(-r / rc)
            / (4.0 * numpy.pi * a * a * r * (1.0 + r / a) ** 2)
        )
        return (
            rho
            * r ** (2.0 * beta)
            * ((2.0 * beta - 1.0) / r - 2.0 / (a + r) - 1.0 / rc)
        )
