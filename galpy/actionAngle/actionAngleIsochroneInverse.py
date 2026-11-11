###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleIsochroneInverse
#
#             Calculate (x,v) coordinates for the Isochrone potential from
#             given actions-angle coordinates
#
###############################################################################
import numpy
from scipy import optimize

from ..backend import get_namespace, is_backend_array, promote_scalars
from ..potential import IsochronePotential
from ..util import conversion
from .actionAngleInverse import actionAngleInverse
from .actionAngleInverseUtil import _ns

# Residual above which an angle is handed to the bracketing solve rather
# than trusted from the Halley iteration.  A module constant so that the
# fallback branch can be exercised: no eccentricity reaches it otherwise.
_KEPLER_RESID_TOL = 1e-12


class actionAngleIsochroneInverse(actionAngleInverse):
    """Inverse action-angle formalism for the isochrone potential, on the Jphi, Jtheta system of Binney & Tremaine (2008); following McGill & Binney (1990) for transformations"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleIsochroneInverse object.

        Parameters
        ----------
        b : float or Quantity, optional
            Scale parameter of the isochrone parameter.
        ip : galpy.potential.IsochronePotential, optional
            Instance of a IsochronePotential.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Either specify b or ip.
        - 2017-11-14 - Started - Bovy (UofT)
        """
        actionAngleInverse.__init__(self, *args, **kwargs)
        if not "b" in kwargs and not "ip" in kwargs:  # pragma: no cover
            raise OSError("Must specify b= for actionAngleIsochrone")
        if "ip" in kwargs:
            ip = kwargs["ip"]
            if not isinstance(ip, IsochronePotential):  # pragma: no cover
                raise OSError(
                    "'Provided ip= does not appear to be an instance of an IsochronePotential"
                )
            # Check the units
            self._pot = ip
            self._check_consistent_units()
            self.b = ip.b
            self.amp = ip._amp
        else:
            self.b = conversion.parse_length(kwargs["b"], ro=self._ro)
            # a backend b (e.g. differentiated) stays on its backend
            _sqrt = (
                get_namespace(self.b).sqrt if is_backend_array(self.b) else numpy.sqrt
            )
            rb = _sqrt(self.b**2.0 + 1.0)
            self.amp = (self.b + rb) ** 2.0 * rb
        # In case we ever decide to implement this in C...
        self._c = False
        ext_loaded = False
        if ext_loaded and (
            ("c" in kwargs and kwargs["c"]) or not "c" in kwargs
        ):  # pragma: no cover
            self._c = True
        else:
            self._c = False
        if not self._c:
            self._ip = IsochronePotential(amp=self.amp, b=self.b)
        # Define _pot, because some functions that use actionAngle instances need this
        self._pot = IsochronePotential(amp=self.amp, b=self.b)
        # Check the units
        self._check_consistent_units()
        return None

    def _evaluate(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) at actions and angles that broadcast against each other (a single torus at a number of angles, or arrays of actions with each point on its own torus).

        Parameters
        ----------
        jr : float
            Radial action.
        jphi : float
            Azimuthal action.
        jz : float
            Vertical action.
        angler : numpy.ndarray
            Radial angle.
        anglephi : numpy.ndarray
            Azimuthal angle.
        anglez : numpy.ndarray
            Vertical angle.

        Returns
        -------
        numpy.ndarray
            Phase-space coordinates [R,vR,vT,z,vz,phi].

        Notes
        -----
        - 2017-11-14 - Written - Bovy (UofT).
        """
        return self._xvFreqs(jr, jphi, jz, angler, anglephi, anglez, **kwargs)[:6]

    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) and the frequencies at actions and angles that broadcast against each other (a single torus at a number of angles, or arrays of actions with each point on its own torus).

        Parameters
        ----------
        jr : float
            Radial action.
        jphi : float
            Azimuthal action.
        jz : float
            Vertical action.
        angler : numpy.ndarray
            Radial angle.
        anglephi : numpy.ndarray
            Azimuthal angle.
        anglez : numpy.ndarray
            Vertical angle.

        Returns
        -------
        tuple
            A tuple containing the phase-space coordinates (R, vR, vT, z, vz, phi), and the frequencies (OmegaR, Omegaphi, Omegaz).

        Notes
        -----
        - 2017-11-15 - Written - Bovy (UofT).
        - 2026-11-11 - Arrays of actions, broadcast against the angles - Bovy (UofT)
        """
        xp = get_namespace(jr, jphi, jz, angler, anglephi, anglez)
        jr, jphi, jz, angler, anglephi, anglez = promote_scalars(
            xp, jr, jphi, jz, angler, anglephi, anglez
        )
        shape = None
        if numpy.ndim(jr) > 0 or numpy.ndim(jphi) > 0 or numpy.ndim(jz) > 0:
            # arrays of actions: everything below is elementwise, so the
            # actions and angles are broadcast to one shape, evaluated flat
            # (each point on its own torus), and returned in that shape
            if xp is numpy:
                args = numpy.broadcast_arrays(
                    *(
                        numpy.atleast_1d(numpy.asarray(t, dtype="float"))
                        for t in (jr, jphi, jz, angler, anglephi, anglez)
                    )
                )
            else:
                args = xp.broadcast_arrays(
                    *(
                        xp.atleast_1d(t)
                        for t in (jr, jphi, jz, angler, anglephi, anglez)
                    )
                )
            shape = args[0].shape
            jr, jphi, jz, angler, anglephi, anglez = (
                xp.reshape(t, (-1,)) for t in args
            )
        L = jz + xp.abs(jphi)  # total angular momentum
        # the torus's (a, e), the latter in the product form that is exact
        # down to the circular orbit
        a, e = self._orbital_params(jr, L)
        ab = a + self.b
        # Solve Kepler's-ish equation eta - (a e/ab) sin(eta) = ar, ar in [0, 2pi)
        angler = (xp.atleast_1d(angler) % (-2.0 * numpy.pi)) % (2.0 * numpy.pi)
        anglephi = xp.atleast_1d(anglephi)
        anglez = xp.atleast_1d(anglez)
        if xp is numpy:
            # Kepler's-ish equation eta - k sin(eta) = angler, solved for all
            # angles at once.  k = a e / (a + b) < 1, so 1 - k cos(eta) >= 1 - k
            # is bounded away from zero: the left-hand side is strictly
            # increasing, the root is unique, and Newton started from the usual
            # eta = angler + k sin(angler) converges for every angle together.
            # Any angle that somehow fails to converge still falls back to the
            # bracketing solve, which cannot fail on 0 <= angler < 2 pi.
            k = a * e / ab
            eta = angler + k * numpy.sin(angler)
            for _ in range(100):
                f = eta - k * numpy.sin(eta) - angler
                fp = 1.0 - k * numpy.cos(eta)
                # Halley rather than Newton: the second derivative is k sin(eta),
                # already to hand, and the cubic convergence removes the slow tail
                # near pericentre at high eccentricity.  Newton needs 41 iterations
                # at k = 0.999 and fails outright at k = 0.99999; Halley needs 7 and
                # 8.  With the spread that small there is nothing to gain from
                # iterating only the unconverged angles -- the bookkeeping costs
                # more than the iterations it saves.
                # k sin(eta) is already implied by f, so the second derivative
                # costs no further transcendental: k sin(eta) = eta - angler - f
                eta -= f / (fp - f * (eta - angler - f) / (2.0 * fp))
                if numpy.all(numpy.fabs(f) < 1e-14 * (1.0 + numpy.fabs(angler))):
                    break
            bad = numpy.fabs(eta - k * numpy.sin(eta) - angler) > _KEPLER_RESID_TOL * (
                1.0 + numpy.fabs(angler)
            )
            for ii in numpy.arange(len(angler))[bad]:
                ar, ki = angler[ii], numpy.broadcast_to(k, angler.shape)[ii]
                eta[ii] = optimize.brentq(
                    lambda x: x - ki * numpy.sin(x) - ar, 0.0, 2.0 * numpy.pi
                )
        else:
            # Differentiable, vectorised bracketed-Newton on [0, 2pi] (f is strictly
            # monotone there since a*e/ab < 1) -- the shared backend root-finder;
            # gradients flow to (jr,jphi,jz) via the implicit-function theorem.
            from ..backend.optimize import brentq as _backend_brentq

            _c = a * e / ab
            eta = _backend_brentq(
                lambda x, c, ar: x - c * xp.sin(x) - ar,
                xp.zeros_like(angler),
                xp.full_like(angler, 2.0 * numpy.pi),
                args=(_c, angler),
            )
        out = self._xvFreqs_from_eta(jr, jphi, jz, eta, anglephi, anglez)
        return out if shape is None else tuple(xp.reshape(o, shape) for o in out)

    def _xvFreqs_from_eta(self, jr, jphi, jz, eta, anglephi, anglez):
        """(x, v) and the frequencies of the (J_r, J_phi, J_z) torus at the
        eccentric anomalies eta and the angles (theta_phi, theta_z): the part
        of _xvFreqs after Kepler's equation, for a caller that knows the
        eccentric anomaly already (the spherical inverse's anomaly map gives
        it directly), with the radial angle taken as eta - k sin(eta)"""
        xp = get_namespace(jr, jphi, jz, eta, anglephi, anglez)
        jr, jphi, jz, eta, anglephi, anglez = promote_scalars(
            xp, jr, jphi, jz, eta, anglephi, anglez
        )
        L = jz + xp.abs(jphi)  # total angular momentum
        L2 = L**2.0
        sqrtfourbkL2 = xp.sqrt(L2 + 4.0 * self.b * self.amp)
        H = self._torus_E(jr, L)
        omegar = (-2.0 * H) ** 1.5 / self.amp
        omegaz = (1.0 + L / sqrtfourbkL2) / 2.0 * omegar
        a, e = self._orbital_params(jr, L)
        ab = a + self.b
        eta = xp.atleast_1d(eta)
        anglephi = xp.atleast_1d(anglephi)
        anglez = xp.atleast_1d(anglez)
        angler = eta - a * e / ab * xp.sin(eta)
        r, vr, _ = self._profile(a, e, eta)
        taneta2 = xp.tan(eta / 2.0)
        if xp is numpy:
            tan11 = numpy.arctan(numpy.sqrt((1.0 + e) / (1.0 - e)) * taneta2)
            tan12 = numpy.arctan(
                numpy.sqrt(
                    (a * (1.0 + e) + 2.0 * self.b) / (a * (1.0 - e) + 2.0 * self.b)
                )
                * taneta2
            )
            tan11 = numpy.where(tan11 < 0.0, tan11 + numpy.pi, tan11)
            tan12 = numpy.where(tan12 < 0.0, tan12 + numpy.pi, tan12)
        else:
            # the arctangents continued through eta = pi without a branch,
            # arctan(k tan(eta/2)) = eta/2 + arctan[(k - 1) t / (1 + k t^2)]
            # with t = tan(eta/2): the backend root at angler = 0 can land a
            # hair below eta = 0, where the branch above would add pi
            def _cont(km1, k):
                return eta / 2.0 + xp.arctan(km1 * taneta2 / (1.0 + k * taneta2**2))

            k1 = xp.sqrt((1.0 + e) / (1.0 - e))
            den2 = a * (1.0 - e) + 2.0 * self.b
            k2 = xp.sqrt((a * (1.0 + e) + 2.0 * self.b) / den2)
            tan11 = _cont(2.0 * e / ((1.0 - e) * (k1 + 1.0)), k1)
            tan12 = _cont(2.0 * a * e / (den2 * (k2 + 1.0)), k2)
        Lambdaeta = tan11 + L / sqrtfourbkL2 * tan12
        psi = anglez - omegaz / omegar * angler + Lambdaeta
        lowerl = xp.sqrt(1.0 - jphi**2.0 / L2)
        sintheta = xp.sin(psi) * lowerl
        costheta = xp.sqrt(1.0 - sintheta**2.0)
        vtheta = L * lowerl * xp.cos(psi) / costheta / r
        R = r * costheta
        z = r * sintheta
        vR = vr * costheta - vtheta * sintheta
        vz = vr * sintheta + vtheta * costheta
        sinu = sintheta / costheta * jphi / L / lowerl
        u = xp.arcsin(sinu)
        u = xp.where(vtheta < 0.0, numpy.pi - u, u)
        phi = anglephi - xp.sign(jphi) * anglez + u
        # For non-inclined orbits, phi == psi
        phi = xp.where(xp.isfinite(phi), phi, psi)
        phi = phi % (2.0 * numpy.pi)
        phi = xp.where(phi < 0.0, phi + 2.0 * numpy.pi, phi)
        return (R, vR, jphi / R, z, vz, phi, omegar, xp.sign(jphi) * omegaz, omegaz)

    def _Freqs(self, jr, jphi, jz, **kwargs):
        """
        Return the frequencies corresponding to a torus

        Parameters
        ----------
        jr : float
            Radial action
        jphi : float
            Azimuthal action
        jz : float
            Vertical action

        Returns
        -------
        tuple
            A tuple of three floats representing the frequencies (OmegaR, Omegaphi, Omegaz)

        Notes
        -----
        - 2017-11-15 - Written - Bovy (UofT).
        """
        xp = get_namespace(jr, jphi, jz)
        jr, jphi, jz = promote_scalars(xp, jr, jphi, jz)
        L = jz + xp.abs(jphi)  # total angular momentum
        sqrtfourbkL2 = xp.sqrt(L**2.0 + 4.0 * self.b * self.amp)
        H = self._torus_E(jr, L)
        # Calculate the frequencies
        omegar = (-2.0 * H) ** 1.5 / self.amp
        omegaz = (1.0 + L / sqrtfourbkL2) / 2.0 * omegar
        return (omegar, xp.sign(jphi) * omegaz, omegaz)

    # ---------- the closed forms of the isochrone's tori, which the spherical
    # and Staeckel inverses lift their tori onto: energy and radial action, the
    # orbital parameters (a, e) with their derivatives along the actions, and
    # the radius, momentum, flux and cumulative radial action in the eccentric
    # anomaly with their partials with respect to (a, e)

    def _torus_E(self, Jr, L):
        """Energy of the auxiliary torus with actions (J_r, L)"""
        xp = _ns(Jr, L, self.amp, self.b)
        CA = 0.5 * (L + xp.sqrt(L**2 + 4.0 * self.amp * self.b))
        return -(self.amp**2) / (2.0 * (Jr + CA) ** 2)

    def _torus_Jr(self, E, L):
        """Radial action of the auxiliary torus with energy E and angular
        momentum L (the inverse of _torus_E)"""
        xp = _ns(E, L, self.amp, self.b)
        CA = 0.5 * (L + xp.sqrt(L**2 + 4.0 * self.amp * self.b))
        return self.amp / xp.sqrt(-2.0 * E) - CA

    def _orbital_params(self, Jr, L):
        """Closed-form (a, e) of the auxiliary torus with actions (J_r, L):
        a + b = (J_r + C_A)^2 / GM, and e^2 = 1 - L^2 (a + b) / (GM a^2)
        evaluated as the product J_r (J_r + s) [GM a + L (J_r + C_A)] /
        (GM a)^2 with s = sqrt(L^2 + 4 GM b), the same quantity with its
        cancellation at small J_r removed, so that e ~ sqrt(J_r) down to
        round-off and vanishes exactly at the circular orbit"""
        xp = _ns(Jr, L, self.amp, self.b)
        GM, b = self.amp, self.b
        s = xp.sqrt(L**2 + 4.0 * GM * b)
        CA = 0.5 * (L + s)
        a = (Jr + CA) ** 2 / GM - b
        e = xp.sqrt(Jr * (Jr + s) * (GM * a + L * (Jr + CA))) / (GM * a)
        return a, e

    def _orbital_param_chains(self, Jr, L, dJr, dL):
        """(a, e) of the auxiliary torus and their derivatives along a
        chain with dJr = dJ_r/dalpha and dL = dL/dalpha, all closed forms"""
        xp = _ns(Jr, L, dJr, dL, self.amp, self.b)
        GM, b = self.amp, self.b
        sq = xp.sqrt(L**2 + 4.0 * GM * b)
        CA = 0.5 * (L + sq)
        EA = -(GM**2) / (2.0 * (Jr + CA) ** 2)
        dEA = GM**2 / (Jr + CA) ** 3 * (dJr + 0.5 * (1.0 + L / sq) * dL)
        a, e = self._orbital_params(Jr, L)
        da = GM / (2.0 * EA**2) * dEA
        de = (
            2.0 * L * dL / (2.0 * EA * a**2)
            - L**2 * (dEA * a + 2.0 * EA * da) / (2.0 * EA**2 * a**3)
        ) / (2.0 * e)
        return a, e, da, de

    def _profile(self, a, e, eta):
        """The auxiliary torus's radius, radial momentum, and dr^A/deta with
        its vanishing factor sin(eta) removed, at eccentric anomaly eta:
        sqrt(b^2 + r^A^2) = b + a (1 - e cos eta), so that with
        y = 1 - e cos eta and beta = b/a, r^A = a sqrt(y (y + 2 beta)),
        p^A = sqrt(GM/(a+b)) a e sin(eta) / r^A and dr^A/deta = g^A sin(eta)
        with g^A = a e (y + beta) / sqrt(y (y + 2 beta)), all closed forms"""
        xp = _ns(a, e, eta, self.amp, self.b)
        beta = self.b / a
        y = 1.0 - e * xp.cos(eta)
        sq = xp.sqrt(y * (y + 2.0 * beta))
        rA = a * sq
        pA = xp.sqrt(self.amp / (a + self.b)) * a * e * xp.sin(eta) / rA
        gA = a * e * (y + beta) / sq
        return rA, pA, gA

    def _radius_partials(self, a, e, eta):
        """Partials of the auxiliary radius with respect to its torus
        parameters (a, e) at fixed eccentric anomaly, from r^A^2 =
        a^2 y (y + 2 b/a) with y = 1 - e cos eta"""
        xp = _ns(a, e, eta, self.b)
        y = 1.0 - e * xp.cos(eta)
        sA = self.b + a * y  # = sqrt(b^2 + r^A^2)
        rA = xp.sqrt(sA**2 - self.b**2)
        return y * sA / rA, -a * sA * xp.cos(eta) / rA

    def _flux_derivs(self, a, e, eta):
        """The auxiliary's radial action flux f^A = p^A dr^A/deta at
        eccentric anomaly eta, and its partials with respect to the torus
        parameters (a, e) at fixed eta, all closed forms: with beta = b/a,
        y = 1 - e cos(eta), c = sqrt(GM/(a+b)) and g(y) = (y + beta) /
        [y (y + 2 beta)], f^A = c a e^2 sin^2(eta) g"""
        xp = _ns(a, e, eta, self.amp, self.b)
        b = self.b
        beta = b / a
        y = 1.0 - e * xp.cos(eta)
        c = xp.sqrt(self.amp / (a + b))
        s2 = xp.sin(eta) ** 2
        g = (y + beta) / (y * (y + 2.0 * beta))
        dg_dy = -(y**2 + 2.0 * beta * y + 2.0 * beta**2) / (y * (y + 2.0 * beta)) ** 2
        dg_da = (b / a**2) / (y + 2.0 * beta) ** 2
        fA = c * a * e**2 * s2 * g
        dfA_da = e**2 * s2 * (c * (a + 2.0 * b) / (2.0 * (a + b)) * g + c * a * dg_da)
        dfA_de = c * a * s2 * (2.0 * e * g - e**2 * dg_dy * xp.cos(eta))
        return fA, dfA_da, dfA_de

    def _orbital_params2(self, Jr, L):
        """The auxiliary torus's (a, e) with their first and second partials
        in (J_r, L), closed forms: a + b = (J_r + C_A)^2/GM with C_A = (L +
        s)/2, s = sqrt(L^2 + 4 GM b), and e^2 = 1 - L^2 F(a) with F = (a +
        b)/(GM a^2), differentiated through a; e itself is the product form
        of _orbital_params, so that the chain to e's partials
        divides by an e that is exact down to the circular edge"""
        xp = _ns(Jr, L, self.amp, self.b)
        GM, b = self.amp, self.b
        s = xp.sqrt(L**2 + 4.0 * GM * b)
        C = 0.5 * (L + s)
        C_L, C_LL = 0.5 * (1.0 + L / s), 2.0 * GM * b / s**3
        a, e = self._orbital_params(Jr, L)
        a_J, a_L = 2.0 * (Jr + C) / GM, 2.0 * (Jr + C) * C_L / GM
        a_JJ, a_JL = 2.0 / GM, 2.0 * C_L / GM
        a_LL = (2.0 * C_L**2 + 2.0 * (Jr + C) * C_LL) / GM
        F = (a + b) / (GM * a**2)
        F_a, F_aa = -(a + 2.0 * b) / (GM * a**3), (2.0 * a + 6.0 * b) / (GM * a**4)
        E2_J = -(L**2) * F_a * a_J
        E2_L = -2.0 * L * F - L**2 * F_a * a_L
        E2_JJ = -(L**2) * (F_aa * a_J**2 + F_a * a_JJ)
        E2_JL = -2.0 * L * F_a * a_J - L**2 * (F_aa * a_J * a_L + F_a * a_JL)
        E2_LL = -2.0 * F - 4.0 * L * F_a * a_L - L**2 * (F_aa * a_L**2 + F_a * a_LL)
        e_J, e_L = E2_J / (2.0 * e), E2_L / (2.0 * e)
        e_JJ = E2_JJ / (2.0 * e) - E2_J**2 / (4.0 * e**3)
        e_JL = E2_JL / (2.0 * e) - E2_J * E2_L / (4.0 * e**3)
        e_LL = E2_LL / (2.0 * e) - E2_L**2 / (4.0 * e**3)
        return a, e, (a_J, a_L, a_JJ, a_JL, a_LL), (e_J, e_L, e_JJ, e_JL, e_LL)

    def _flux_eta(self, a, e, eta):
        """The auxiliary's radial action flux f^A at eccentric anomaly eta
        and its derivative with respect to eta: with beta = b/a, y = 1 - e
        cos(eta), c = sqrt(GM/(a+b)) and g(y) = (y + beta)/[y (y + 2 beta)],
        f^A = c a e^2 sin^2(eta) g and df^A/deta = c a e^2 sin(eta) [2
        cos(eta) g + e sin^2(eta) g'(y)]"""
        xp = _ns(a, e, eta, self.amp, self.b)
        b = self.b
        beta = b / a
        y = 1.0 - e * xp.cos(eta)
        c = xp.sqrt(self.amp / (a + b))
        g = (y + beta) / (y * (y + 2.0 * beta))
        dg_dy = -(y**2 + 2.0 * beta * y + 2.0 * beta**2) / (y * (y + 2.0 * beta)) ** 2
        se, ce = xp.sin(eta), xp.cos(eta)
        return c * a * e**2 * se**2 * g, c * a * e**2 * se * (
            2.0 * ce * g + e * se**2 * dg_dy
        )

    def _cumulative2(self, a, e, eta):
        """The auxiliary's cumulative radial action A^A(eta) with its first
        and second partials with respect to the torus parameters (a, e) at
        fixed eta, the gathered closed form of _cumulative
        differentiated once more: A^A = (c a) B with B = e sin(eta) +
        lambda eta - sum_A s_A phi_A, where s_A = sqrt(A^2 - e^2), phi_A =
        arctan(k_A t) - arctan(t) with t = tan(eta/2) and k_A = sqrt((A +
        e)/(A - e)), lambda = sum_A (A - s_A)/2, A = 1 and A = 1 + 2 b/a;
        the a-dependence enters through c a, the second A, and nothing
        else, and every partial of phi_A goes through dphi_A/dk_A = t/(1 +
        k_A^2 t^2) and its k-derivative. Verified against mpmath to 5e-13."""
        xp = _ns(a, e, eta, self.amp, self.b)
        b = self.b
        if xp is numpy:
            eta = numpy.asarray(eta, dtype="float")
        c = xp.sqrt(self.amp / (a + b))
        ca = c * a
        ca_a = c * (a + 2.0 * b) / (2.0 * (a + b))
        ca_aa = -c * (a + 4.0 * b) / (4.0 * (a + b) ** 2)
        t = xp.tan(0.5 * eta)
        A2 = 1.0 + 2.0 * b / a
        A2_a, A2_aa = -2.0 * b / a**2, 4.0 * b / a**3
        B, B_e, B_a = e * xp.sin(eta), xp.sin(eta), 0.0 * eta
        B_ee, B_ae, B_aa = 0.0 * eta, 0.0 * eta, 0.0 * eta
        lam = lam_e = lam_ee = lam_a = lam_ae = lam_aa = 0.0
        for A, dA, d2A in ((1.0, 0.0, 0.0), (A2, A2_a, A2_aa)):
            s = xp.sqrt(A**2 - e**2)
            s_e, s_ee = -e / s, -1.0 / s - e**2 / s**3
            s_A, s_AA, s_eA = A / s, -(e**2) / s**3, A * e / s**3
            lam += 0.5 * (A - s)
            lam_e += 0.5 * e / s
            lam_ee += 0.5 * (1.0 / s + e**2 / s**3)
            lam_a += 0.5 * dA * (1.0 - A / s)
            lam_ae += -0.5 * dA * A * e / s**3
            lam_aa += 0.5 * (d2A * (1.0 - A / s) + dA**2 * e**2 / s**3)
            q = (A + e) / (A - e)
            k = xp.sqrt(q)
            q_e, q_A = 2.0 * A / (A - e) ** 2, -2.0 * e / (A - e) ** 2
            q_ee = 4.0 * A / (A - e) ** 3
            q_eA = -2.0 * (A + e) / (A - e) ** 3
            q_AA = 4.0 * e / (A - e) ** 3
            k_e, k_A = q_e / (2.0 * k), q_A / (2.0 * k)
            k_ee = q_ee / (2.0 * k) - q_e**2 / (4.0 * k**3)
            k_eA = q_eA / (2.0 * k) - q_e * q_A / (4.0 * k**3)
            k_AA = q_AA / (2.0 * k) - q_A**2 / (4.0 * k**3)
            km1 = 2.0 * e / ((A - e) * (k + 1.0))
            phi = xp.arctan(km1 * t / (1.0 + k * t**2))
            phi_k = t / (1.0 + k**2 * t**2)
            phi_kk = -2.0 * k * t**3 / (1.0 + k**2 * t**2) ** 2
            phi_e, phi_A = phi_k * k_e, phi_k * k_A
            phi_ee = phi_kk * k_e**2 + phi_k * k_ee
            phi_eA = phi_kk * k_e * k_A + phi_k * k_eA
            phi_AA = phi_kk * k_A**2 + phi_k * k_AA
            T = s * phi
            T_e, T_A = s_e * phi + s * phi_e, s_A * phi + s * phi_A
            T_ee = s_ee * phi + 2.0 * s_e * phi_e + s * phi_ee
            T_eA = s_eA * phi + s_e * phi_A + s_A * phi_e + s * phi_eA
            T_AA = s_AA * phi + 2.0 * s_A * phi_A + s * phi_AA
            B = B - T
            B_e = B_e - T_e
            B_a = B_a - T_A * dA
            B_ee = B_ee - T_ee
            B_ae = B_ae - T_eA * dA
            B_aa = B_aa - (T_AA * dA**2 + T_A * d2A)
        B = B + lam * eta
        B_e = B_e + lam_e * eta
        B_a = B_a + lam_a * eta
        B_ee = B_ee + lam_ee * eta
        B_ae = B_ae + lam_ae * eta
        B_aa = B_aa + lam_aa * eta
        return (
            ca * B,
            ca_a * B + ca * B_a,
            ca * B_e,
            ca_aa * B + 2.0 * ca_a * B_a + ca * B_aa,
            ca_a * B_e + ca * B_ae,
            ca * B_ee,
        )

    def _cumulative(self, a, e, eta):
        """The auxiliary's cumulative radial action from pericentre,
        A^A(eta) = int_0^eta f^A, and its partials with respect to the torus
        parameters (a, e) at fixed eta, all closed forms: with beta = b/a and
        c = sqrt(GM/(a+b)), the flux's factor (y + beta)/[y (y + 2 beta)] is
        [1/y + 1/(y + 2 beta)]/2, and each term integrates to
        c a [e sin(eta)/2 + A eta/2 - sqrt(A^2 - e^2) Theta_A(eta)] with A = 1
        and A = 1 + 2 beta, where Theta_A = arctan(sqrt((A+e)/(A-e))
        tan(eta/2)), continued through its branches by the winding number of
        eta -- the two arctangents of the isochrone's angle relation. The
        secular rate over a libration is the auxiliary's radial action.

        The sum is O(eta^3) at pericentre while its terms are O(eta), and
        their cancellation would leave the round-off of the largest term,
        (1 + beta) eta, on a cumulative action e^2 times smaller; the anomaly
        that the matching's Newton finds on this function inherits that
        error divided by the flux, and the map's derivative amplifies it by
        the number of harmonics. So the linear parts are gathered
        analytically: Theta_A = eta/2 + phi_A with phi_A = arctan(k_A t) -
        arctan(t) = arctan[(k_A - 1) t/(1 + k_A t^2)] (t = tan(eta/2), no
        branch to continue), the coefficient of eta that remains, lambda =
        1 + beta - sum_A sqrt(A^2 - e^2)/2 = e^2/2 sum_A 1/(A + sqrt(A^2 -
        e^2)), and k_A - 1 = 2 e/[(A - e)(k_A + 1)] are formed without
        cancellation, and every term is O(e)."""
        xp = _ns(a, e, eta, self.amp, self.b)
        b = self.b
        c = xp.sqrt(self.amp / (a + b))
        beta = b / a
        if xp is numpy:
            eta = numpy.asarray(eta, dtype="float")
        t = xp.tan(0.5 * eta)
        dca_da = c * (a + 2.0 * b) / (2.0 * (a + b))
        dbeta_da = -b / a**2
        bracket = e * xp.sin(eta)
        dbr_de = xp.sin(eta)
        dbr_da = xp.zeros_like(eta)
        lam, dlam_de, dlam_da = 0.0, 0.0, 0.0
        for A, dA_da in ((1.0, 0.0), (1.0 + 2.0 * beta, 2.0 * dbeta_da)):
            k = xp.sqrt((A + e) / (A - e))
            sq = xp.sqrt(A**2 - e**2)
            km1 = 2.0 * e / ((A - e) * (k + 1.0))
            phi = xp.arctan(km1 * t / (1.0 + k * t**2))
            dphi_dk = t / (1.0 + k**2 * t**2)
            dk_de = A / (k * (A - e) ** 2)
            dk_dA = -e / (k * (A - e) ** 2)
            lam = lam + 0.5 * e**2 / (A + sq)
            dlam_de = dlam_de + 0.5 * e / sq
            dlam_da = dlam_da - 0.5 * e**2 * dA_da / (sq * (A + sq))
            bracket = bracket - sq * phi
            dbr_de = dbr_de + (e / sq) * phi - sq * dphi_dk * dk_de
            dbr_da = dbr_da - (A / sq) * dA_da * phi - sq * dphi_dk * dk_dA * dA_da
        bracket = bracket + lam * eta
        dbr_de = dbr_de + dlam_de * eta
        dbr_da = dbr_da + dlam_da * eta
        return c * a * bracket, dca_da * bracket + c * a * dbr_da, c * a * dbr_de
