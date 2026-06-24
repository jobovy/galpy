###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleSpherical
#
#      methods:
#             __call__: returns (jr,lz,jz)
#             actionsFreqs: returns (jr,lz,jz,Or,Op,Oz)
#             actionsFreqsAngles: returns (jr,lz,jz,Or,Op,Oz,ar,ap,az)
#
###############################################################################
import copy

import numpy
from scipy import integrate, optimize

from ..backend import get_namespace, is_backend_array
from ..potential import _dim, epifreq, omegac, vcirc
from ..potential.planarPotential import _evaluateplanarPotentials
from ..potential.Potential import (
    _check_potential_list_and_deprecate,
    _evaluatePotentials,
)
from ..util import quadpack
from .actionAngle import UnboundError, actionAngle

_EPS = 10.0**-15.0


class actionAngleSpherical(actionAngle):
    """Action-angle formalism for spherical potentials"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleSpherical object.

        Parameters
        ----------
        pot : Potential or a combined potential formed using addition (pot1+pot2+…)
            A spherical potential.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        _gamma : float, optional
            Replace Lz by Lz+gamma Jz in effective potential when using this class as part of actionAngleAdiabatic (internal use).

        Notes
        -----
        - 2013-12-28 - Written - Bovy (IAS)
        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
        if not "pot" in kwargs:  # pragma: no cover
            raise OSError("Must specify pot= for actionAngleSpherical")
        self._pot = _check_potential_list_and_deprecate(kwargs["pot"])
        # Also store a 'planar' (2D) version of the potential, only potential
        # used in this class
        if _dim(self._pot) == 2:
            self._2dpot = self._pot
        else:
            self._2dpot = self._pot.toPlanar()
        # The following for if we ever implement this code in C
        self._c = False
        ext_loaded = False
        if ext_loaded and (("c" in kwargs and kwargs["c"]) or not "c" in kwargs):
            self._c = True  # pragma: no cover
        else:
            self._c = False
        # gamma for when we use this as part of the adiabatic approx.
        self._gamma = kwargs.get("_gamma", 0.0)
        # Check the units
        self._check_consistent_units()
        return None

    def _evaluate(self, *args, **kwargs):
        """
        Evaluate the actions (jr,lz,jz).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        fixed_quad: bool, optional
            if True, use n=10 fixed_quad integration
        **kwargs: dict, optional
            scipy.integrate.quadrature or .fixed_quad keywords

        Returns
        -------
        tuple
            (jr,lz,jz)

        Notes
        -----
        - 2013-12-28 - Written - Bovy (IAS)
        """
        fixed_quad = kwargs.pop("fixed_quad", False)
        extra_Jz = kwargs.pop("_Jz", None)
        if len(args) == 5:  # R,vR.vT, z, vz
            R, vR, vT, z, vz = args
        elif len(args) == 6:  # R,vR.vT, z, vz, phi
            R, vR, vT, z, vz, phi = args
        else:
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        if self._c:  # pragma: no cover
            pass
        elif is_backend_array(R):
            # jax/torch inputs: vectorised, differentiable path. Detected on R
            # alone (all coords share a backend); numpy/Quantity stays below.
            r, vr, vt, E, L, Lz, L2 = self._setup_backend(R, vR, vT, z, vz, extra_Jz)
            rperi, rap = self._calc_rperi_rap_backend(r, vr, vt, E, L)
            Jr = self._calc_jr_backend(rperi, rap, E, L)
            xp = get_namespace(R)
            return (Jr, Lz, L - xp.abs(Lz))
        else:
            r = numpy.sqrt(R**2.0 + z**2.0)
            vr = (R * vR + z * vz) / r
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            E = (
                _evaluateplanarPotentials(self._2dpot, r)
                + vR**2.0 / 2.0
                + vT**2.0 / 2.0
                + vz**2.0 / 2.0
            )
            L = numpy.sqrt(L2)
            vt = L / r
            if self._gamma != 0.0 and not extra_Jz is None:
                L += self._gamma * extra_Jz
                E += L**2.0 / 2.0 / r**2.0 - vt**2.0 / 2.0
            # Actions
            Jphi = Lz
            Jz = L - numpy.fabs(Lz)
            # Jr requires some more work
            Jr = []
            for ii in range(len(r)):
                rperi, rap = self._calc_rperi_rap(r[ii], vr[ii], vt[ii], E[ii], L[ii])
                Jr.append(self._calc_jr(rperi, rap, E[ii], L[ii], fixed_quad, **kwargs))
            return (numpy.array(Jr), Jphi, Jz)

    def _actionsFreqs(self, *args, **kwargs):
        """
        Evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        fixed_quad: bool, optional
            if True, use n=10 fixed_quad integration
        **kwargs: dict, optional
            scipy.integrate.quadrature or .fixed_quad keywords

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)

        Notes
        -----
        - 2013-12-28 - Written - Bovy (IAS)
        """
        fixed_quad = kwargs.pop("fixed_quad", False)
        extra_Jz = kwargs.pop("_Jz", None)
        if len(args) == 5:  # R,vR.vT, z, vz
            R, vR, vT, z, vz = args
        elif len(args) == 6:  # R,vR.vT, z, vz, phi
            R, vR, vT, z, vz, phi = args
        else:
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        if self._c:  # pragma: no cover
            pass
        elif is_backend_array(R):
            # jax/torch inputs: vectorised, differentiable path (see _evaluate).
            return self._actionsFreqs_backend(R, vR, vT, z, vz, extra_Jz)
        else:
            r = numpy.sqrt(R**2.0 + z**2.0)
            vr = (R * vR + z * vz) / r
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            E = (
                _evaluateplanarPotentials(self._2dpot, r)
                + vR**2.0 / 2.0
                + vT**2.0 / 2.0
                + vz**2.0 / 2.0
            )
            L = numpy.sqrt(L2)
            vt = L / r
            if self._gamma != 0.0 and not extra_Jz is None:
                L += self._gamma * extra_Jz
                E += L**2.0 / 2.0 / r**2.0 - vt**2.0 / 2.0
            # Actions
            Jphi = Lz
            Jz = L - numpy.fabs(Lz)
            # Jr requires some more work
            Jr = []
            Or = []
            Op = []
            for ii in range(len(r)):
                rperi, rap = self._calc_rperi_rap(r[ii], vr[ii], vt[ii], E[ii], L[ii])
                Jr.append(self._calc_jr(rperi, rap, E[ii], L[ii], fixed_quad, **kwargs))
                # Radial period
                if Jr[-1] < 10.0**-9.0:  # Circular orbit
                    Or.append(epifreq(self._2dpot, r[ii], use_physical=False))
                    Op.append(omegac(self._2dpot, r[ii], use_physical=False))
                    continue
                Rmean = (
                    numpy.exp((numpy.log(rperi) + numpy.log(rap)) / 2.0)
                    if rperi > 0.0
                    else rap / 2.0
                )
                Or.append(
                    self._calc_or(Rmean, rperi, rap, E[ii], L[ii], fixed_quad, **kwargs)
                )
                Op.append(
                    self._calc_op(
                        Or[-1], Rmean, rperi, rap, E[ii], L[ii], fixed_quad, **kwargs
                    )
                )
            Op = numpy.array(Op)
            Oz = copy.copy(Op)
            Op[vT < 0.0] *= -1.0
            return (numpy.array(Jr), Jphi, Jz, numpy.array(Or), Op, Oz)

    def _actionsFreqsAngles(self, *args, **kwargs):
        """
        Evaluate the actions, frequencies, and angles (jr,lz,jz,Omegar,Omegaphi,Omegaz,ar,aphi,az).

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        fixed_quad: bool, optional
            if True, use n=10 fixed_quad integration
        **kwargs: dict, optional
            scipy.integrate.quadrature or .fixed_quad keywords

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,ar,aphi,az)

        Notes
        -----
        - 2013-12-29 - Written - Bovy (IAS)
        """
        fixed_quad = kwargs.pop("fixed_quad", False)
        extra_Jz = kwargs.pop("_Jz", None)
        if len(args) == 5:  # R,vR.vT, z, vz pragma: no cover
            raise OSError("You need to provide phi when calculating angles")
        elif len(args) == 6:  # R,vR.vT, z, vz, phi
            R, vR, vT, z, vz, phi = args
        else:
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
            phi = self._eval_phi
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
            phi = numpy.array([phi])
        if self._c:  # pragma: no cover
            pass
        elif is_backend_array(R):
            # jax/torch inputs: vectorised, differentiable path (see _evaluate).
            return self._actionsFreqsAngles_backend(R, vR, vT, z, vz, phi, extra_Jz)
        else:
            r = numpy.sqrt(R**2.0 + z**2.0)
            vr = (R * vR + z * vz) / r
            vtheta = (z * vR - R * vz) / r
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            E = (
                _evaluateplanarPotentials(self._2dpot, r)
                + vR**2.0 / 2.0
                + vT**2.0 / 2.0
                + vz**2.0 / 2.0
            )
            L = numpy.sqrt(L2)
            vt = L / r
            if self._gamma != 0.0 and not extra_Jz is None:
                L += self._gamma * extra_Jz
                E += L**2.0 / 2.0 / r**2.0 - vt**2.0 / 2.0
            # Actions
            Jphi = Lz
            Jz = L - numpy.fabs(Lz)
            # Jr requires some more work
            Jr = []
            Or = []
            Op = []
            ar = []
            az = []
            # Calculate the longitude of the ascending node
            asc = self._calc_long_asc(z, R, vtheta, phi, Lz, L)
            for ii in range(len(r)):
                rperi, rap = self._calc_rperi_rap(r[ii], vr[ii], vt[ii], E[ii], L[ii])
                Jr.append(self._calc_jr(rperi, rap, E[ii], L[ii], fixed_quad, **kwargs))
                # Radial period
                Rmean = (
                    numpy.exp((numpy.log(rperi) + numpy.log(rap)) / 2.0)
                    if rperi > 0
                    else rap / 2.0
                )
                if Jr[-1] < 10.0**-9.0:  # Circular orbit
                    Or.append(epifreq(self._2dpot, r[ii], use_physical=False))
                    Op.append(omegac(self._2dpot, r[ii], use_physical=False))
                else:
                    Or.append(
                        self._calc_or(
                            Rmean, rperi, rap, E[ii], L[ii], fixed_quad, **kwargs
                        )
                    )
                    Op.append(
                        self._calc_op(
                            Or[-1],
                            Rmean,
                            rperi,
                            rap,
                            E[ii],
                            L[ii],
                            fixed_quad,
                            **kwargs,
                        )
                    )
                # Angles
                ar.append(
                    self._calc_angler(
                        Or[-1],
                        r[ii],
                        Rmean,
                        rperi,
                        rap,
                        E[ii],
                        L[ii],
                        vr[ii],
                        fixed_quad,
                        **kwargs,
                    )
                )
                az.append(
                    self._calc_anglez(
                        Or[-1],
                        Op[-1],
                        ar[-1],
                        z[ii],
                        r[ii],
                        Rmean,
                        rperi,
                        rap,
                        E[ii],
                        L[ii],
                        Lz[ii],
                        vr[ii],
                        vtheta[ii],
                        phi[ii],
                        fixed_quad,
                        **kwargs,
                    )
                )
            Op = numpy.array(Op)
            Oz = copy.copy(Op)
            Op[vT < 0.0] *= -1.0
            ap = copy.copy(asc)
            ar = numpy.array(ar)
            az = numpy.array(az)
            ap[vT < 0.0] -= az[vT < 0.0]
            ap[vT >= 0.0] += az[vT >= 0.0]
            ar = ar % (2.0 * numpy.pi)
            ap = ap % (2.0 * numpy.pi)
            az = az % (2.0 * numpy.pi)
            return (numpy.array(Jr), Jphi, Jz, numpy.array(Or), Op, Oz, ar, ap, az)

    def _EccZmaxRperiRap(self, *args, **kwargs):
        """
        Evaluate the eccentricity, maximum height above the plane, peri- and apocenter for a spherical potential.

        Parameters
        ----------
        *args : tuple
            Either:
            a) R,vR,vT,z,vz[,phi]:
                1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
            b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument

        Returns
        -------
        tuple
            (e,zmax,rperi,rap)

        Notes
        -----
        - 2017-12-22 - Written - Bovy (UofT)
        """
        extra_Jz = kwargs.pop("_Jz", None)
        if len(args) == 5:  # R,vR.vT, z, vz
            R, vR, vT, z, vz = args
        elif len(args) == 6:  # R,vR.vT, z, vz, phi
            R, vR, vT, z, vz, phi = args
        else:
            self._parse_eval_args(*args)
            R = self._eval_R
            vR = self._eval_vR
            vT = self._eval_vT
            z = self._eval_z
            vz = self._eval_vz
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        if self._c:  # pragma: no cover
            pass
        elif is_backend_array(R):
            # jax/torch inputs: vectorised, differentiable path (see _evaluate).
            xp = get_namespace(R)
            r, vr, vt, E, L, Lz, L2 = self._setup_backend(R, vR, vT, z, vz, extra_Jz)
            rperi, rap = self._calc_rperi_rap_backend(r, vr, vt, E, L)
            return (
                (rap - rperi) / (rap + rperi),
                rap * xp.sqrt(1.0 - Lz**2.0 / L2),
                rperi,
                rap,
            )
        else:
            r = numpy.sqrt(R**2.0 + z**2.0)
            vr = (R * vR + z * vz) / r
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            L = numpy.sqrt(L2)
            E = (
                _evaluateplanarPotentials(self._2dpot, r)
                + vR**2.0 / 2.0
                + vT**2.0 / 2.0
                + vz**2.0 / 2.0
            )
            vt = L / r
            if self._gamma != 0.0 and not extra_Jz is None:
                L += self._gamma * extra_Jz
                E += L**2.0 / 2.0 / r**2.0 - vt**2.0 / 2.0
            rperi, rap = [], []
            for ii in range(len(r)):
                trperi, trap = self._calc_rperi_rap(r[ii], vr[ii], vt[ii], E[ii], L[ii])
                rperi.append(trperi)
                rap.append(trap)
            rperi = numpy.array(rperi)
            rap = numpy.array(rap)
            return (
                (rap - rperi) / (rap + rperi),
                rap * numpy.sqrt(1.0 - Lz**2.0 / L2),
                rperi,
                rap,
            )

    # ------------------------------------------------------------------ backend
    # Vectorised, differentiable (jax/torch) implementations of the per-object
    # setup + rperi/rap root-find + Jr quadrature, shared by _evaluate and
    # _EccZmaxRperiRap. The numpy path is untouched (byte-identical); these run
    # only when the inputs are backend arrays. PR-1 scope: gamma==0 only.

    def _setup_backend(self, R, vR, vT, z, vz, extra_Jz):
        """Backend (jax/torch) version of the shared setup arithmetic.

        Identical to the numpy preamble but namespace-agnostic (xp.* / xp.abs),
        which is byte-identical on numpy. Returns (r, vr, vt, E, L, Lz, L2).
        """
        if self._gamma != 0.0:
            raise NotImplementedError(
                "actionAngleSpherical backend (jax/torch) path supports only "
                "_gamma==0 (standalone use); the adiabatic _gamma!=0 case is not "
                "yet implemented in-backend"
            )
        xp = get_namespace(R)
        r = xp.sqrt(R**2.0 + z**2.0)
        vr = (R * vR + z * vz) / r
        Lz = R * vT
        Lx = -z * vT
        Ly = z * vR - R * vz
        L2 = Lx * Lx + Ly * Ly + Lz * Lz
        E = (
            _evaluateplanarPotentials(self._2dpot, r)
            + vR**2.0 / 2.0
            + vT**2.0 / 2.0
            + vz**2.0 / 2.0
        )
        L = xp.sqrt(L2)
        vt = L / r
        return (r, vr, vt, E, L, Lz, L2)

    def _calc_rperi_rap_backend(self, r, vr, vt, E, L):
        """Vectorised rperi/rap via the shared backend bracketed root-finder.

        Brackets are found by a fixed-schedule expanding search (no scipy
        while-loop) so the whole batch runs in lockstep; the special cases
        (circular / exact peri- or apocenter / plunge through r=0) are handled
        by dead-branch-guarded xp.where overrides. gamma==0 (=> startsign=+1).
        """
        from ..backend.optimize import brentq as _backend_brentq

        xp = get_namespace(r)

        def f(R_, E_, L_):
            return (
                E_
                - _evaluateplanarPotentials(self._2dpot, R_)
                - L_**2.0 / 2.0 / R_**2.0
            )

        # Fixed-schedule bracketing (mirrors _rapRperiAxiFindStart, vectorised):
        # halve from r/2 until f<=0 (or below the floor) for rperi's lower end,
        # double from 2r until f<=0 for rap's upper end. 80 steps >> any needed.
        rstart = r / 2.0
        for _ in range(80):
            rstart = xp.where(
                (f(rstart, E, L) > 0.0) & (rstart > 1e-9), rstart / 2.0, rstart
            )
        rend = 2.0 * r
        for _ in range(80):
            rend = xp.where(f(rend, E, L) > 0.0, rend * 2.0, rend)
        # Special cases (all are vr==0, measure-zero among generic test orbits).
        vcirc_r = vcirc(self._2dpot, r, use_physical=False)
        is_circ = (vr == 0.0) & (xp.abs(vt - vcirc_r) < _EPS)
        at_peri = (vr == 0.0) & (vt > vcirc_r)
        at_apo = (vr == 0.0) & (vt < vcirc_r)
        plunge = rstart <= 1e-9
        # Widen degenerate brackets to a safe [r/2, r] BEFORE brentq (dead-branch
        # guard: the where below overrides these elements, no NaN-poison). For
        # exact peri/apo, f(r)==0 at the shared endpoint, so nudge it off the
        # root (mirrors scipy's rperi+1e-5 / rap-1e-6 nudges).
        rstart_safe = xp.where(plunge, r / 2.0, rstart)
        rperi_hi = xp.where(at_apo, r - 1e-6, r)
        rap_lo = xp.where(at_peri, r + 1e-5, r)
        rperi = _backend_brentq(f, rstart_safe, rperi_hi, args=(E, L))
        rap = _backend_brentq(f, rap_lo, rend, args=(E, L))
        rperi = xp.where(is_circ | at_peri, r, xp.where(plunge, 0.0, rperi))
        rap = xp.where(is_circ | at_apo, r, rap)
        return (rperi, rap)

    def _calc_jr_backend(self, rperi, rap, E, L):
        """Vectorised, differentiable Jr = (1/pi) int_rperi^rap sqrt(...) dr.

        Substitute r = rperi + (rap-rperi) sin^2(theta), theta in [0, pi/2], so
        the sqrt's endpoint zeros are absorbed by the 2 sin cos Jacobian and the
        integrand is smooth. Fixed-order Gauss-Legendre via the shared
        backend.quadrature.fixed_quad (n=25). The radicand is clipped >=0 before
        the sqrt (sqrt'(0)=inf would NaN-poison reverse-mode AD).
        """
        from ..backend.quadrature import fixed_quad

        xp = get_namespace(rperi)
        span = rap - rperi

        def integrand(theta):
            # theta: (n,) node array; build r(theta): (N, n).
            sin = xp.sin(theta)[None, :]
            cos = xp.cos(theta)[None, :]
            rr = rperi[:, None] + span[:, None] * sin**2.0
            Phi = _evaluateplanarPotentials(self._2dpot, rr)
            rad = 2.0 * (E[:, None] - Phi) - L[:, None] ** 2.0 / rr**2.0
            rad = xp.where(rad > 0.0, rad, 0.0)  # clip before sqrt (AD guard)
            return xp.sqrt(rad) * span[:, None] * 2.0 * sin * cos

        Jr = fixed_quad(xp, integrand, 0.0, numpy.pi / 2.0, n=25)
        return Jr / numpy.pi

    # -------------------------------------------------- backend freqs + angles
    # Vectorised, differentiable (jax/torch) Or/Op (radial+azimuthal frequency)
    # and ar/ap/az (angles), mirroring the per-object numpy _calc_or/_calc_op/
    # _calc_angler/_calc_anglez/_calc_long_asc. The two t^2-substituted panels
    # of each period/angle integral are evaluated with backend.quadrature.
    # fixed_quad on a fixed [0, 1] panel: the per-object upper limit `lim`
    # (sqrt(Rmean-rperi) etc.) is folded INTO the integrand via t = lim*s
    # (dt = lim ds), so fixed_quad's scalar a, b stay 0, 1 while `lim` is a
    # shape-(N,) array. The 2t Jacobian of the substitution cancels the
    # 1/sqrt endpoint zero; the radicand is clipped >=0 before the sqrt
    # (sqrt'(0)=inf would NaN-poison reverse-mode AD), with the unused panel
    # (lim==0) contributing exactly 0. The numpy path is untouched.

    def _panel_backend(self, xp, base, sign, lim, E, L, azimuthal):
        """One t^2-substituted Gauss-Legendre panel of a period/angle integral.

        Integrates ``2 t / _JrSphericalIntegrand(r) [/ r**2 if azimuthal]`` over
        ``t in [0, lim]`` with ``r = base + sign * t**2`` (sign=+1 small panel
        with base=rperi, sign=-1 large panel with base=rap). The fixed [0, 1]
        GL panel uses ``t = lim * s`` so the per-object ``lim`` (shape (N,))
        multiplies inside the integrand (dt = lim ds) and fixed_quad's limits
        stay scalar. ``lim`` is float (==0 makes this panel contribute 0).
        """
        from ..backend.quadrature import fixed_quad

        def integrand(s):  # s: (n,) GL nodes -> (N, n)
            t = lim[:, None] * s[None, :]
            rr = base[:, None] + sign * t**2.0
            Phi = _evaluateplanarPotentials(self._2dpot, rr)
            rad = 2.0 * (E[:, None] - Phi) - L[:, None] ** 2.0 / rr**2.0
            # clip before sqrt (AD guard); the masked-out (rad<=0) endpoint sits
            # where 2t->0 anyway, so a 0 there is harmless.
            rad = xp.where(rad > 0.0, rad, xp.ones_like(rad))
            val = 2.0 * t / xp.sqrt(rad)
            if azimuthal:
                val = val / rr**2.0
            return val * lim[:, None]  # dt = lim ds

        return fixed_quad(xp, integrand, 0.0, 1.0, n=25)

    def _calc_or_op_backend(self, Rmean, rperi, rap, E, L):
        """Vectorised Or (radial freq) and Op (azimuthal freq magnitude).

        Tr = 2*(small panel [0, sqrt(Rmean-rperi)] + large panel
        [0, sqrt(rap-Rmean)]) of 2t/_Jr; Or = 2pi/Tr. The same panels weighted
        by 1/r**2 give I; Op = 2*L*I * Or / (2 pi). Returns (Or, Op) with Op the
        positive magnitude (the vT<0 sign flip is applied by the caller).
        """
        xp = get_namespace(rperi)
        limS = xp.sqrt(xp.where(Rmean > rperi, Rmean - rperi, xp.zeros_like(Rmean)))
        limL = xp.sqrt(xp.where(rap > Rmean, rap - Rmean, xp.zeros_like(Rmean)))
        Tr = 2.0 * (
            self._panel_backend(xp, rperi, 1.0, limS, E, L, False)
            + self._panel_backend(xp, rap, -1.0, limL, E, L, False)
        )
        Or = 2.0 * numpy.pi / Tr
        I = (
            2.0
            * L
            * (
                self._panel_backend(xp, rperi, 1.0, limS, E, L, True)
                + self._panel_backend(xp, rap, -1.0, limL, E, L, True)
            )
        )
        Op = I * Or / 2.0 / numpy.pi
        return (Or, Op)

    def _calc_angler_backend(self, Or, r, Rmean, rperi, rap, E, L, vr):
        """Vectorised radial angle ar (un-modded; caller takes % 2pi).

        Mirrors _calc_angler: if r<Rmean integrate the small panel to
        sqrt(r-rperi) and (vr<0) wr=2pi-wr; else integrate the large panel to
        sqrt(rap-r) and wr = pi+wr (vr<0) / pi-wr (vr>=0).
        """
        xp = get_namespace(r)
        limS = xp.sqrt(xp.where(r > rperi, r - rperi, xp.zeros_like(r)))
        limL = xp.sqrt(xp.where(rap > r, rap - r, xp.zeros_like(r)))
        wr_small = Or * self._panel_backend(xp, rperi, 1.0, limS, E, L, False)
        wr_small = xp.where(vr < 0.0, 2.0 * numpy.pi - wr_small, wr_small)
        wr_large = Or * self._panel_backend(xp, rap, -1.0, limL, E, L, False)
        wr_large = xp.where(vr < 0.0, numpy.pi + wr_large, numpy.pi - wr_large)
        return xp.where(r < Rmean, wr_small, wr_large)

    def _calc_anglez_backend(
        self, Or, Op, ar, z, r, Rmean, rperi, rap, E, L, Lz, vr, vtheta, phi
    ):
        """Vectorised vertical angle az (un-modded; caller takes % 2pi).

        Mirrors _calc_anglez: psi from sinpsi=z/r/sin(inclination) (clipped,
        vtheta>0 -> pi-psi, non-inclined -> phi), then wz = L*I-integral to the
        same data-dependent limit (vr quadrant fixes via dpsi=Op/Or*2pi), and
        az = -wz + psi + Op/Or*ar (ar un-modded). Op here is the magnitude.
        """
        xp = get_namespace(r)
        # psi (inclination phase)
        i_incl = xp.arccos(xp.where(xp.abs(Lz / L) < 1.0, Lz / L, xp.sign(Lz / L)))
        sini = xp.sin(i_incl)
        sini_safe = xp.where(sini == 0.0, xp.ones_like(sini), sini)
        sinpsi = z / r / sini_safe
        finite = xp.isfinite(sinpsi)
        sinpsi_c = xp.where(
            sinpsi > 1.0,
            xp.ones_like(sinpsi),
            xp.where(sinpsi < -1.0, -xp.ones_like(sinpsi), sinpsi),
        )
        psi = xp.arcsin(sinpsi_c)
        psi = xp.where(vtheta > 0.0, numpy.pi - psi, psi)
        psi = xp.where(finite, psi, phi)  # non-inclined: psi=phi
        psi = psi % (2.0 * numpy.pi)
        dpsi = Op / Or * 2.0 * numpy.pi  # full I integral
        limS = xp.sqrt(xp.where(r > rperi, r - rperi, xp.zeros_like(r)))
        limL = xp.sqrt(xp.where(rap > r, rap - r, xp.zeros_like(r)))
        wz_small = L * self._panel_backend(xp, rperi, 1.0, limS, E, L, True)
        wz_small = xp.where(vr < 0.0, dpsi - wz_small, wz_small)
        wz_large = L * self._panel_backend(xp, rap, -1.0, limL, E, L, True)
        wz_large = xp.where(vr < 0.0, dpsi / 2.0 + wz_large, dpsi / 2.0 - wz_large)
        wz = xp.where(r < Rmean, wz_small, wz_large)
        return -wz + psi + Op / Or * ar

    def _calc_long_asc_backend(self, z, R, vtheta, phi, Lz, L):
        """Vectorised longitude of the ascending node (mirror _calc_long_asc)."""
        xp = get_namespace(R)
        i = xp.arccos(Lz / L)
        sinu = z / R / xp.tan(i)
        sinu = xp.where(
            (sinu > 1.0) & (sinu < 1.0 + 10.0**-7.0), xp.ones_like(sinu), sinu
        )
        sinu = xp.where((sinu < -1.0) & xp.isfinite(sinu), -xp.ones_like(sinu), sinu)
        sinu_c = xp.where(
            sinu > 1.0,
            xp.ones_like(sinu),
            xp.where(sinu < -1.0, -xp.ones_like(sinu), sinu),
        )
        u = xp.arcsin(sinu_c)
        u = xp.where(vtheta > 0.0, numpy.pi - u, u)
        u = xp.where(xp.isfinite(u), u, phi)  # non-inclined: Omega=0 (u=phi)
        return phi - u

    def _actionsFreqs_backend(self, R, vR, vT, z, vz, extra_Jz):
        """Vectorised (Jr,Lz,Jz,Or,Op,Oz) for backend (jax/torch) inputs."""
        xp = get_namespace(R)
        r, vr, vt, E, L, Lz, L2 = self._setup_backend(R, vR, vT, z, vz, extra_Jz)
        rperi, rap = self._calc_rperi_rap_backend(r, vr, vt, E, L)
        Jr = self._calc_jr_backend(rperi, rap, E, L)
        Jphi = Lz
        Jz = L - xp.abs(Lz)
        # Rmean = exp((log rperi + log rap)/2) if rperi>0 else rap/2 (guard log)
        rperi_safe = xp.where(rperi > 0.0, rperi, xp.ones_like(rperi))
        Rmean = xp.where(
            rperi > 0.0,
            xp.exp((xp.log(rperi_safe) + xp.log(rap)) / 2.0),
            rap / 2.0,
        )
        Or, Op = self._calc_or_op_backend(Rmean, rperi, rap, E, L)
        # Circular branch (Jr<1e-9): epifreq/omegac (backend-ready forces).
        is_circ = Jr < 10.0**-9.0
        Or = xp.where(is_circ, epifreq(self._2dpot, r, use_physical=False), Or)
        Op = xp.where(is_circ, omegac(self._2dpot, r, use_physical=False), Op)
        Oz = Op  # copy (magnitude)
        Op = xp.where(vT < 0.0, -Op, Op)
        return (Jr, Jphi, Jz, Or, Op, Oz)

    def _actionsFreqsAngles_backend(self, R, vR, vT, z, vz, phi, extra_Jz):
        """Vectorised (Jr,Lz,Jz,Or,Op,Oz,ar,ap,az) for backend inputs."""
        xp = get_namespace(R)
        r, vr, vt, E, L, Lz, L2 = self._setup_backend(R, vR, vT, z, vz, extra_Jz)
        vtheta = (z * vR - R * vz) / r
        rperi, rap = self._calc_rperi_rap_backend(r, vr, vt, E, L)
        Jr = self._calc_jr_backend(rperi, rap, E, L)
        Jphi = Lz
        Jz = L - xp.abs(Lz)
        rperi_safe = xp.where(rperi > 0.0, rperi, xp.ones_like(rperi))
        Rmean = xp.where(
            rperi > 0.0,
            xp.exp((xp.log(rperi_safe) + xp.log(rap)) / 2.0),
            rap / 2.0,
        )
        Or, Op = self._calc_or_op_backend(Rmean, rperi, rap, E, L)
        is_circ = Jr < 10.0**-9.0
        Or = xp.where(is_circ, epifreq(self._2dpot, r, use_physical=False), Or)
        Op = xp.where(is_circ, omegac(self._2dpot, r, use_physical=False), Op)
        # Angles (ar, az un-modded; Op is the magnitude here, as in numpy).
        asc = self._calc_long_asc_backend(z, R, vtheta, phi, Lz, L)
        ar = self._calc_angler_backend(Or, r, Rmean, rperi, rap, E, L, vr)
        az = self._calc_anglez_backend(
            Or, Op, ar, z, r, Rmean, rperi, rap, E, L, Lz, vr, vtheta, phi
        )
        Oz = Op  # copy (magnitude)
        Op = xp.where(vT < 0.0, -Op, Op)
        ap = xp.where(vT < 0.0, asc - az, asc + az)
        ar = ar % (2.0 * numpy.pi)
        ap = ap % (2.0 * numpy.pi)
        az = az % (2.0 * numpy.pi)
        return (Jr, Jphi, Jz, Or, Op, Oz, ar, ap, az)

    def _calc_rperi_rap(self, r, vr, vt, E, L):
        if (
            vr == 0.0
            and numpy.fabs(vt - vcirc(self._2dpot, r, use_physical=False)) < _EPS
        ):
            # We are on a circular orbit
            rperi = r
            rap = r
        elif vr == 0.0 and vt > vcirc(self._2dpot, r, use_physical=False):
            # We are exactly at pericenter
            rperi = r
            if self._gamma != 0.0:
                startsign = _rapRperiAxiEq(r + 10.0**-8.0, E, L, self._2dpot)
                startsign /= numpy.fabs(startsign)
            else:
                startsign = 1.0
            rend = _rapRperiAxiFindStart(
                r, E, L, self._2dpot, rap=True, startsign=startsign
            )
            rap = optimize.brentq(
                _rapRperiAxiEq, rperi + 0.00001, rend, args=(E, L, self._2dpot)
            )
        elif vr == 0.0 and vt < vcirc(self._2dpot, r, use_physical=False):
            # We are exactly at apocenter
            rap = r
            if self._gamma != 0.0:
                startsign = _rapRperiAxiEq(r - 10.0**-8.0, E, L, self._2dpot)
                startsign /= numpy.fabs(startsign)
            else:
                startsign = 1.0
            rstart = _rapRperiAxiFindStart(r, E, L, self._2dpot, startsign=startsign)
            if rstart == 0.0:
                rperi = 0.0
            else:
                rperi = optimize.brentq(
                    _rapRperiAxiEq, rstart, rap - 0.000001, args=(E, L, self._2dpot)
                )
        else:
            if self._gamma != 0.0:
                startsign = _rapRperiAxiEq(r, E, L, self._2dpot)
                startsign /= numpy.fabs(startsign)
            else:
                startsign = 1.0
            rstart = _rapRperiAxiFindStart(r, E, L, self._2dpot, startsign=startsign)
            if rstart == 0.0:
                rperi = 0.0
            else:
                try:
                    rperi = optimize.brentq(
                        _rapRperiAxiEq, rstart, r, (E, L, self._2dpot), maxiter=200
                    )
                except RuntimeError:  # pragma: no cover
                    raise UnboundError("Orbit seems to be unbound")
            rend = _rapRperiAxiFindStart(
                r, E, L, self._2dpot, rap=True, startsign=startsign
            )
            rap = optimize.brentq(_rapRperiAxiEq, r, rend, (E, L, self._2dpot))
        return (rperi, rap)

    def _calc_jr(self, rperi, rap, E, L, fixed_quad, **kwargs):
        if fixed_quad:
            return (
                integrate.fixed_quad(
                    _JrSphericalIntegrand,
                    rperi,
                    rap,
                    args=(E, L, self._2dpot),
                    n=10,
                    **kwargs,
                )[0]
                / numpy.pi
            )
        else:
            return (
                numpy.array(
                    integrate.quad(
                        _JrSphericalIntegrand,
                        rperi,
                        rap,
                        args=(E, L, self._2dpot),
                        **kwargs,
                    )
                )
            )[0] / numpy.pi

    def _calc_or(self, Rmean, rperi, rap, E, L, fixed_quad, **kwargs):
        Tr = 0.0
        if Rmean > rperi and not fixed_quad:
            Tr += numpy.array(
                quadpack.quadrature(
                    _TrSphericalIntegrandSmall,
                    0.0,
                    numpy.sqrt(Rmean - rperi),
                    args=(E, L, self._2dpot, rperi),
                    **kwargs,
                )
            )[0]
        elif Rmean > rperi and fixed_quad:
            Tr += integrate.fixed_quad(
                _TrSphericalIntegrandSmall,
                0.0,
                numpy.sqrt(Rmean - rperi),
                args=(E, L, self._2dpot, rperi),
                n=10,
                **kwargs,
            )[0]
        if Rmean < rap and not fixed_quad:
            Tr += numpy.array(
                quadpack.quadrature(
                    _TrSphericalIntegrandLarge,
                    0.0,
                    numpy.sqrt(rap - Rmean),
                    args=(E, L, self._2dpot, rap),
                    **kwargs,
                )
            )[0]
        elif Rmean < rap and fixed_quad:
            Tr += integrate.fixed_quad(
                _TrSphericalIntegrandLarge,
                0.0,
                numpy.sqrt(rap - Rmean),
                args=(E, L, self._2dpot, rap),
                n=10,
                **kwargs,
            )[0]
        Tr = 2.0 * Tr
        return 2.0 * numpy.pi / Tr

    def _calc_op(self, Or, Rmean, rperi, rap, E, L, fixed_quad, **kwargs):
        # Azimuthal period
        I = 0.0
        if Rmean > rperi and not fixed_quad:
            I += numpy.array(
                quadpack.quadrature(
                    _ISphericalIntegrandSmall,
                    0.0,
                    numpy.sqrt(Rmean - rperi),
                    args=(E, L, self._2dpot, rperi),
                    **kwargs,
                )
            )[0]
        elif Rmean > rperi and fixed_quad:
            I += integrate.fixed_quad(
                _ISphericalIntegrandSmall,
                0.0,
                numpy.sqrt(Rmean - rperi),
                args=(E, L, self._2dpot, rperi),
                n=10,
                **kwargs,
            )[0]
        if Rmean < rap and not fixed_quad:
            I += numpy.array(
                quadpack.quadrature(
                    _ISphericalIntegrandLarge,
                    0.0,
                    numpy.sqrt(rap - Rmean),
                    args=(E, L, self._2dpot, rap),
                    **kwargs,
                )
            )[0]
        elif Rmean < rap and fixed_quad:
            I += integrate.fixed_quad(
                _ISphericalIntegrandLarge,
                0.0,
                numpy.sqrt(rap - Rmean),
                args=(E, L, self._2dpot, rap),
                n=10,
                **kwargs,
            )[0]
        I *= 2 * L
        return I * Or / 2.0 / numpy.pi

    def _calc_long_asc(self, z, R, vtheta, phi, Lz, L):
        i = numpy.arccos(Lz / L)
        sinu = z / R / numpy.tan(i)
        pindx = (sinu > 1.0) * (sinu < (1.0 + 10.0**-7.0))
        sinu[pindx] = 1.0
        pindx = (sinu < -1.0) * numpy.isfinite(sinu)
        sinu[pindx] = -1.0
        u = numpy.arcsin(sinu)
        vzindx = vtheta > 0.0
        u[vzindx] = numpy.pi - u[vzindx]
        # For non-inclined orbits, we set Omega=0 by convention
        u[True ^ numpy.isfinite(u)] = phi[True ^ numpy.isfinite(u)]
        return phi - u

    def _calc_angler(self, Or, r, Rmean, rperi, rap, E, L, vr, fixed_quad, **kwargs):
        if r < Rmean:
            if r > rperi and not fixed_quad:
                wr = (
                    Or
                    * quadpack.quadrature(
                        _TrSphericalIntegrandSmall,
                        0.0,
                        numpy.sqrt(r - rperi),
                        args=(E, L, self._2dpot, rperi),
                        **kwargs,
                    )[0]
                )
            elif r > rperi and fixed_quad:
                wr = (
                    Or
                    * integrate.fixed_quad(
                        _TrSphericalIntegrandSmall,
                        0.0,
                        numpy.sqrt(r - rperi),
                        args=(E, L, self._2dpot, rperi),
                        n=10,
                        **kwargs,
                    )[0]
                )
            else:
                wr = 0.0
            if vr < 0.0:
                wr = 2 * numpy.pi - wr
        else:
            if r < rap and not fixed_quad:
                wr = (
                    Or
                    * quadpack.quadrature(
                        _TrSphericalIntegrandLarge,
                        0.0,
                        numpy.sqrt(rap - r),
                        args=(E, L, self._2dpot, rap),
                        **kwargs,
                    )[0]
                )
            elif r < rap and fixed_quad:
                wr = (
                    Or
                    * integrate.fixed_quad(
                        _TrSphericalIntegrandLarge,
                        0.0,
                        numpy.sqrt(rap - r),
                        args=(E, L, self._2dpot, rap),
                        n=10,
                        **kwargs,
                    )[0]
                )
            else:
                wr = 0.0
            if vr < 0.0:
                wr = numpy.pi + wr
            else:
                wr = numpy.pi - wr
        return wr

    def _calc_anglez(
        self,
        Or,
        Op,
        ar,
        z,
        r,
        Rmean,
        rperi,
        rap,
        E,
        L,
        Lz,
        vr,
        vtheta,
        phi,
        fixed_quad,
        **kwargs,
    ):
        # First calculate psi
        i = numpy.arccos(Lz / L)
        sinpsi = z / r / numpy.sin(i)
        if numpy.isfinite(sinpsi):
            sinpsi = 1.0 if sinpsi > 1.0 else (-1.0 if sinpsi < -1.0 else sinpsi)
            psi = numpy.arcsin(sinpsi)
            if vtheta > 0.0:
                psi = numpy.pi - psi
        else:
            psi = phi
        psi = psi % (2.0 * numpy.pi)
        # Calculate dSr/dL
        dpsi = Op / Or * 2.0 * numpy.pi  # this is the full I integral
        if r < Rmean:
            if numpy.sqrt(r - rperi) == 0.0:
                wz = 0.0
            elif not fixed_quad:
                wz = (
                    L
                    * quadpack.quadrature(
                        _ISphericalIntegrandSmall,
                        0.0,
                        numpy.sqrt(r - rperi),
                        args=(E, L, self._2dpot, rperi),
                        **kwargs,
                    )[0]
                )
            elif fixed_quad:
                wz = (
                    L
                    * integrate.fixed_quad(
                        _ISphericalIntegrandSmall,
                        0.0,
                        numpy.sqrt(r - rperi),
                        args=(E, L, self._2dpot, rperi),
                        n=10,
                        **kwargs,
                    )[0]
                )
            if vr < 0.0:
                wz = dpsi - wz
        else:
            if numpy.sqrt(rap - r) == 0.0:
                wz = 0.0
            elif not fixed_quad:
                wz = (
                    L
                    * quadpack.quadrature(
                        _ISphericalIntegrandLarge,
                        0.0,
                        numpy.sqrt(rap - r),
                        args=(E, L, self._2dpot, rap),
                        **kwargs,
                    )[0]
                )
            elif fixed_quad:
                wz = (
                    L
                    * integrate.fixed_quad(
                        _ISphericalIntegrandLarge,
                        0.0,
                        numpy.sqrt(rap - r),
                        args=(E, L, self._2dpot, rap),
                        n=10,
                        **kwargs,
                    )[0]
                )
            if vr < 0.0:
                wz = dpsi / 2.0 + wz
            else:
                wz = dpsi / 2.0 - wz
        # Add everything
        wz = -wz + psi + Op / Or * ar
        return wz


def _JrSphericalIntegrand(r, E, L, pot):
    """The J_r integrand"""
    return numpy.sqrt(2.0 * (E - _evaluateplanarPotentials(pot, r)) - L**2.0 / r**2.0)


def _TrSphericalIntegrandSmall(t, E, L, pot, rperi):
    r = rperi + t**2.0  # part of the transformation
    return 2.0 * t / _JrSphericalIntegrand(r, E, L, pot)


def _TrSphericalIntegrandLarge(t, E, L, pot, rap):
    r = rap - t**2.0  # part of the transformation
    return 2.0 * t / _JrSphericalIntegrand(r, E, L, pot)


def _ISphericalIntegrandSmall(t, E, L, pot, rperi):
    r = rperi + t**2.0  # part of the transformation
    return 2.0 * t / _JrSphericalIntegrand(r, E, L, pot) / r**2.0


def _ISphericalIntegrandLarge(t, E, L, pot, rap):
    r = rap - t**2.0  # part of the transformation
    return 2.0 * t / _JrSphericalIntegrand(r, E, L, pot) / r**2.0


def _rapRperiAxiEq(R, E, L, pot):
    """The vr=0 equation that needs to be solved to find apo- and pericenter"""
    return E - _evaluateplanarPotentials(pot, R) - L**2.0 / 2.0 / R**2.0


def _rapRperiAxiFindStart(R, E, L, pot, rap=False, startsign=1.0):
    """
    Find adequate start or end points to solve for rap and rperi

    Parameters
    ----------
    R : float
        Galactocentric radius
    E : float
        energy
    L : float
        angular momentum
    pot : Potential object or a combined potential formed using addition (pot1+pot2+…)
        Potential
    rap : bool, optional
        if True, find the rap end-point (default is False)
    startsign : float, optional
        set to -1 if the function is not positive (due to gamma in the modified adiabatic approximation) (default is 1.0)

    Returns
    -------
    float
        rstart or rend

    Notes
    -----
    - 2010-12-01 - Written - Bovy (NYU)
    """
    if rap:
        rtry = 2.0 * R
    else:
        rtry = R / 2.0
    while startsign * _rapRperiAxiEq(rtry, E, L, pot) > 0.0 and rtry > 0.000000001:
        if rap:
            if rtry > 100.0:  # pragma: no cover
                raise UnboundError("Orbit seems to be unbound")
            rtry *= 2.0
        else:
            rtry /= 2.0
    if rtry < 0.000000001:
        return 0.0
    return rtry
