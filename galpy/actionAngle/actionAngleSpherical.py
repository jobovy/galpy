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
import warnings
from collections import namedtuple

import numpy
from scipy import integrate, optimize

from ..potential import _dim, epifreq, rl, vcirc
from ..potential.planarPotential import (
    _evaluateplanarPotentials,
    _evaluateplanarRforces,
)
from ..potential.Potential import (
    PotentialError,
    _check_potential_list_and_deprecate,
    _evaluateRforces,
)
from ..util import quadpack
from .actionAngle import UnboundError, actionAngle

_EPS = 10.0**-15.0
# an orbit is not close to circular, and never reaches the circular-orbit
# machinery, when its radial velocity or its tangential velocity's excess over
# the local circular speed exceeds this fraction of that speed (an epicycle of
# relative half-width w / r_c has |v_r| / v_c <= 2 w / r_c and
# |v_t / v_c(r) - 1| ~ w / r_c)
_NEARCIRC = 0.05
# an orbit is small, and has its radial problem solved relative to the point
# itself, when its energy above the effective potential's minimum (harmonic
# estimate from the point's radial velocity and effective force) is below this
# fraction of the effective potential there: the plain energy difference then
# has few digits left, and the force is smooth across the whole orbit
_SMALL = 10.0**-4.0
# the turning points are solved for to this relative tolerance
_XTOL = 10.0**-14.0
# below this harmonic half-width relative to the circular radius, an orbit is
# an epicycle: its action and frequencies are the epicycle's, accurate to
# (w / r_c)^2, its angles and turning points too, accurate to w / r_c. Above
# it the general path's quadratures, relative to the circular orbit, are more
# accurate for a potential whose force is evaluated to round-off; for one
# whose force loses digits (the Burkert and NFW forces at radii well inside
# their scale) the quadratures' accuracy is that round-off, amplified next to
# the turning points as the libration shrinks
_EPICYCLE = 10.0**-5.0
# the order of the relative problem's Gaussian quadratures is capped, so that
# their nodes stay clear of the round-off zone next to the turning points (a
# force that loses digits keeps successive orders from agreeing, and the
# innermost node of a high order lands in that zone); the integrands, with
# the turning points' square roots substituted away, converge well before
_RELATIVE_MAXITER = 20


class _RelativeEffectivePotential:
    """The effective potential Phi(r) + L^2 / (2 r^2) of the angular momentum
    L relative to its value at a reference radius r_0 (the circular orbit of
    an orbit close to circular, the point itself of a small orbit), free of
    the cancellation between two energies of the order of the potential's
    that limits the general path there: within a window around r_0 the
    potential's difference is a Gauss-Legendre quadrature of the radial
    force from r_0 (a smooth integrand, exact to round-off), and the
    centrifugal term's difference is in closed form; outside the window the
    direct difference is accurate. Stands in for the planar potential in the
    radial equation and in the quadratures of the general path, with the
    energy above Phi_eff(r_0) in place of the energy. Once the turning
    points have been found, the radicand is anchored to vanish exactly at
    them by subtracting the linear interpolant of its residuals there (a
    correction at round-off), so that the quadratures next to a turning
    point see the radicand's leading behaviour rather than a residual whose
    sign is random."""

    _x, _w = numpy.polynomial.legendre.leggauss(6)

    def __init__(self, pot, force, r0, L, window):
        # pot: the planar potential; force: the radial force at an array of
        # radii in the plane (through the three-dimensional potential with an
        # array of zero heights when there is one: some potentials' forces
        # stack their coordinates and want them all of one shape); window:
        # |r - r_0| within which the force is integrated
        self._pot, self._force, self._r0, self._L2 = pot, force, r0, L**2.0
        self._window = window
        self._Phi0 = _evaluateplanarPotentials(pot, r0)
        self._anchor = None

    def anchor(self, E, rperi, rap):
        """Anchor the radicand 2 [E - Phi_eff] to vanish exactly at the
        turning points found"""
        self._anchor = None
        # the centre is not a turning point: a radial orbit's pericentre
        self._anchor = (
            rperi,
            rap,
            E - self(rperi) if rperi > 0.0 else 0.0,
            E - self(rap),
        )

    def __call__(self, r):
        scalar = numpy.ndim(r) == 0
        r = numpy.atleast_1d(numpy.asarray(r, dtype="float"))
        d = r - self._r0
        dPhi = numpy.empty_like(d)
        near = numpy.fabs(d) < self._window
        if numpy.any(near):
            dn = d[near]
            sn = self._r0 + dn[:, None] * (self._x[None, :] + 1.0) / 2.0
            F = self._force(sn.ravel()).reshape(sn.shape)
            dPhi[near] = -0.5 * dn * (F @ self._w)
        if not numpy.all(near):
            dPhi[~near] = _evaluateplanarPotentials(self._pot, r[~near]) - self._Phi0
        out = dPhi
        if self._L2 > 0.0:
            out = out - self._L2 * d * (r + self._r0) / (2.0 * r**2.0 * self._r0**2.0)
        if self._anchor is not None:
            rperi, rap, dp, da = self._anchor
            out = out + (dp * (rap - r) + da * (r - rperi)) / (rap - rperi)
        return out[0] if scalar else out


# the circular orbit an orbit close to circular is an epicycle around: its
# radius, the epicycle and circular frequencies, the energy above it, the
# harmonic half-width, and the effective potential relative to it
_Epicycle = namedtuple("_Epicycle", ["rc", "kappa", "Omc", "dE", "w", "relpot"])


def _quadrature(pot, func, a, b, args=(), **kwargs):
    """The fixed-tolerance Gaussian quadrature of the general path; for the
    relative problem with its order capped and the cap's warning silenced
    (the accuracy is then the potential's own round-off)"""
    if isinstance(pot, _RelativeEffectivePotential):
        kwargs = {"maxiter": _RELATIVE_MAXITER, **kwargs}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", quadpack.AccuracyWarning)
            return quadpack.quadrature(func, a, b, args=args, **kwargs)
    return quadpack.quadrature(func, a, b, args=args, **kwargs)


def _is_epicycle(epi, threshold=None):
    threshold = _EPICYCLE if threshold is None else threshold
    return epi is not None and epi.w < threshold * epi.rc


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
                epi, Eu, potu, w = self._problem(r[ii], vr[ii], E[ii], L[ii])
                if _is_epicycle(epi):
                    Jr.append(epi.dE / epi.kappa)
                    continue
                rperi, rap = self._calc_rperi_rap(
                    r[ii], vr[ii], Eu, L[ii], w=w, pot=potu
                )
                Jr.append(
                    self._calc_jr(rperi, rap, Eu, L[ii], fixed_quad, pot=potu, **kwargs)
                )
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
                epi, Eu, potu, w = self._problem(r[ii], vr[ii], E[ii], L[ii])
                if _is_epicycle(epi):
                    Jr.append(epi.dE / epi.kappa)
                    Or.append(epi.kappa)
                    Op.append(epi.Omc)
                    continue
                rperi, rap = self._calc_rperi_rap(
                    r[ii], vr[ii], Eu, L[ii], w=w, pot=potu
                )
                Jr.append(
                    self._calc_jr(rperi, rap, Eu, L[ii], fixed_quad, pot=potu, **kwargs)
                )
                # Radial period
                Rmean = (
                    numpy.exp((numpy.log(rperi) + numpy.log(rap)) / 2.0)
                    if rperi > 0.0
                    else rap / 2.0
                )
                Or.append(
                    self._calc_or(
                        Rmean, rperi, rap, Eu, L[ii], fixed_quad, pot=potu, **kwargs
                    )
                )
                Op.append(
                    self._calc_op(
                        Or[-1],
                        Rmean,
                        rperi,
                        rap,
                        Eu,
                        L[ii],
                        fixed_quad,
                        pot=potu,
                        **kwargs,
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
                epi, Eu, potu, w = self._problem(r[ii], vr[ii], E[ii], L[ii])
                if _is_epicycle(epi):
                    # an epicycle: r = r_c - w cos(theta_r), v_r = w kappa
                    # sin(theta_r), and the azimuth runs ahead of its angle
                    # by (2 Omega_c / kappa)(w / r_c) sin(theta_r)
                    Jr.append(epi.dE / epi.kappa)
                    Or.append(epi.kappa)
                    Op.append(epi.Omc)
                    ar.append(numpy.arctan2(vr[ii] / epi.kappa, epi.rc - r[ii]))
                    az.append(
                        self._calc_psi(z[ii], r[ii], L[ii], Lz[ii], vtheta[ii], phi[ii])
                        - 2.0 * epi.Omc / epi.kappa * epi.w / epi.rc * numpy.sin(ar[-1])
                    )
                    continue
                rperi, rap = self._calc_rperi_rap(
                    r[ii], vr[ii], Eu, L[ii], w=w, pot=potu
                )
                Jr.append(
                    self._calc_jr(rperi, rap, Eu, L[ii], fixed_quad, pot=potu, **kwargs)
                )
                # Radial period
                Rmean = (
                    numpy.exp((numpy.log(rperi) + numpy.log(rap)) / 2.0)
                    if rperi > 0
                    else rap / 2.0
                )
                Or.append(
                    self._calc_or(
                        Rmean, rperi, rap, Eu, L[ii], fixed_quad, pot=potu, **kwargs
                    )
                )
                Op.append(
                    self._calc_op(
                        Or[-1],
                        Rmean,
                        rperi,
                        rap,
                        Eu,
                        L[ii],
                        fixed_quad,
                        pot=potu,
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
                        Eu,
                        L[ii],
                        vr[ii],
                        fixed_quad,
                        pot=potu,
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
                        Eu,
                        L[ii],
                        Lz[ii],
                        vr[ii],
                        vtheta[ii],
                        phi[ii],
                        fixed_quad,
                        pot=potu,
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
                epi, Eu, potu, w = self._problem(r[ii], vr[ii], E[ii], L[ii])
                if _is_epicycle(epi):
                    # a circular orbit to round-off has its radius as both
                    # turning points, exactly
                    trperi, trap = (
                        (r[ii], r[ii])
                        if epi.w <= _EPS * epi.rc
                        else (epi.rc - epi.w, epi.rc + epi.w)
                    )
                else:
                    trperi, trap = self._calc_rperi_rap(
                        r[ii], vr[ii], Eu, L[ii], w=w, pot=potu
                    )
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

    def _epicycle(self, r, vr, E, L):
        """The circular orbit an orbit close to circular is an epicycle
        around, or None when the orbit is not close to circular. First a
        screen on the forces alone (the radial velocity and the tangential
        velocity's excess over the local circular speed, relative to it), so
        that eccentric orbits, radial orbits, and potentials without second
        derivatives never reach the circular-orbit machinery; then the
        circular radius r_c(L) polished to round-off by Newton on the
        effective force (whose slope there is kappa^2), the epicycle
        frequency kappa (from the force's finite difference when the
        potential has no second derivative), the circular frequency
        L / r_c^2, the effective potential relative to the circular orbit,
        the energy above the circular orbit's dE = v_r^2 / 2 + [Phi_eff(r) -
        Phi_eff(r_c)], and the harmonic half-width w = sqrt([r - r_c]^2 +
        [v_r / kappa]^2), all free of the cancellation between energies"""
        vc = vcirc(self._2dpot, r, use_physical=False)
        if (
            L == 0.0
            or numpy.fabs(vr) > _NEARCIRC * vc
            or numpy.fabs(L / r - vc) > _NEARCIRC * vc
        ):
            return None
        rc = rl(self._2dpot, L, use_physical=False)

        def _feff(x):  # d Phi_eff / dr
            return -_evaluateplanarRforces(self._2dpot, x) - L**2.0 / x**3.0

        def _kappa(x):  # kappa^2 is the slope of the effective force
            try:
                return epifreq(self._2dpot, x, use_physical=False)
            except PotentialError:  # no second derivative: difference the force
                h = 10.0**-4.0 * x
                return numpy.sqrt((_feff(x + h) - _feff(x - h)) / (2.0 * h))

        kappa = _kappa(rc)
        for _ in range(2):
            rc -= _feff(rc) / kappa**2.0
        kappa = _kappa(rc)
        relpot = _RelativeEffectivePotential(
            self._2dpot, self._radial_force(), rc, L, 0.1 * rc
        )
        dE = max(0.5 * vr**2.0 + relpot(r), 0.0)
        w = numpy.sqrt((r - rc) ** 2.0 + (vr / kappa) ** 2.0)
        return _Epicycle(rc, kappa, L / rc**2.0, dE, w, relpot)

    def _radial_force(self):
        """The radial force in the plane at an array of radii, through the
        planar potential; a force that stacks its coordinates and wants them
        all of one shape (the ellipsoidal potentials) rejects the planar
        wrapper's scalar height, and is evaluated through the wrapped
        three-dimensional potential with an array of zero heights instead
        (which other forces, branching on a scalar height, reject in turn)"""
        pots = self._2dpot if isinstance(self._2dpot, list) else [self._2dpot]

        def _force(x):
            out = 0.0
            for pp in pots:
                try:
                    out = out + _evaluateplanarRforces(pp, x)
                except ValueError:
                    out = out + _evaluateRforces(pp._Pot, x, 0.0 * x)
            return out

        return _force

    def _small_orbit(self, r, vr, L):
        """A small orbit, one whose energy above the effective potential's
        minimum (the harmonic estimate v_r^2 / 2 + f^2 / 2 kappa_eff^2 from
        the point's radial velocity, effective force f and its slope
        kappa_eff^2, by finite differences of the force) is a tiny fraction
        of the effective potential there, has its radial problem solved
        relative to the point itself: the energy above Phi_eff(r) is
        v_r^2 / 2 exactly, and the force is smooth across the whole orbit.
        Returns (v_r^2 / 2, the relative effective potential, the harmonic
        half-width) or None when the orbit is not small (or the effective
        potential is not convex at the point, where the estimate fails)"""
        force = self._radial_force()

        def _feff(x):  # d Phi_eff / dr
            return -force(x) - (L**2.0 / x**3.0 if L > 0.0 else 0.0)

        f = _feff(r)
        h = 10.0**-4.0 * r
        k2 = (_feff(r + h) - _feff(r - h)) / (2.0 * h)
        if not k2 > 0.0:
            return None
        dE = 0.5 * vr**2.0 + f**2.0 / (2.0 * k2)
        Phieff = _evaluateplanarPotentials(self._2dpot, r) + L**2.0 / (2.0 * r**2.0)
        if not dE < _SMALL * numpy.fabs(Phieff):
            return None
        return (
            0.5 * vr**2.0,
            _RelativeEffectivePotential(self._2dpot, force, r, L, r),
            numpy.sqrt(2.0 * dE / k2),
        )

    def _problem(self, r, vr, E, L):
        """The radial problem an orbit's general path solves: for an orbit
        close to circular, the energy above the circular orbit's and the
        effective potential relative to it, with the epicycle's half-width
        for the bracket offsets; for a small orbit, the radial kinetic energy
        and the effective potential relative to the point itself; otherwise
        the energy and the potential"""
        epi = self._epicycle(r, vr, E, L)
        if epi is not None:
            return epi, epi.dE, epi.relpot, epi.w
        small = self._small_orbit(r, vr, L)
        if small is not None:
            return None, small[0], small[1], small[2]
        return None, E, self._2dpot, None

    def _calc_psi(self, z, r, L, Lz, vtheta, phi):
        """The angle in the orbital plane from the ascending node"""
        i = numpy.arccos(Lz / L)
        sinpsi = z / r / numpy.sin(i)
        if numpy.isfinite(sinpsi):
            sinpsi = 1.0 if sinpsi > 1.0 else (-1.0 if sinpsi < -1.0 else sinpsi)
            psi = numpy.arcsin(sinpsi)
            if vtheta > 0.0:
                psi = numpy.pi - psi
        else:
            psi = phi
        return psi % (2.0 * numpy.pi)

    def _calc_rperi_rap(self, r, vr, E, L, w=None, pot=None):
        pot = self._2dpot if pot is None else pot
        # at a turning point when the radial kinetic energy is at round-off
        # (a circular orbit never reaches this: it is an epicycle of zero
        # amplitude, handled before the turning points are looked for); a
        # turning point is the pericenter when the effective potential of
        # the angular momentum L that enters the radial equation -- the
        # adiabatic approximation's gamma modifies it -- rises inward there,
        # that is, when L / r exceeds the circular speed
        scale = (
            E  # the energy above the circular orbit's
            if isinstance(pot, _RelativeEffectivePotential)
            else 0.5 * L**2.0 / r**2.0 + numpy.fabs(E)
        )
        at_turning = 0.5 * vr**2.0 <= _EPS * scale
        # the probe for the sign of the radial equation next to a turning
        # point (the adiabatic approximation's gamma can turn it) and the
        # bracket offsets inside the libration scale with the radius, and
        # with the libration's half-width when that is known
        probe = 10.0**-8.0 * r if w is None else min(10.0**-8.0 * r, 0.1 * w)
        if at_turning and L / r >= vcirc(self._2dpot, r, use_physical=False):
            # We are exactly at pericenter
            rperi = r
            if self._gamma != 0.0:
                startsign = _rapRperiAxiEq(r + probe, E, L, pot)
                startsign /= numpy.fabs(startsign)
            else:
                startsign = 1.0
            rend = _rapRperiAxiFindStart(r, E, L, pot, rap=True, startsign=startsign)
            # an offset inside the libration: a fraction of the epicyclic
            # half-width, which the actual width always exceeds
            delta = 10.0**-5.0 * r if w is None else min(10.0**-5.0 * r, 0.1 * w)
            rap = optimize.brentq(
                _rapRperiAxiEq, rperi + delta, rend, args=(E, L, pot), xtol=_XTOL * r
            )
        elif at_turning:
            # We are exactly at apocenter
            rap = r
            if self._gamma != 0.0:
                startsign = _rapRperiAxiEq(r - probe, E, L, pot)
                startsign /= numpy.fabs(startsign)
            else:
                startsign = 1.0
            rstart = _rapRperiAxiFindStart(r, E, L, pot, startsign=startsign)
            if rstart == 0.0:
                rperi = 0.0
            else:
                delta = 10.0**-6.0 * r if w is None else min(10.0**-6.0 * r, 0.1 * w)
                rperi = optimize.brentq(
                    _rapRperiAxiEq,
                    rstart,
                    rap - delta,
                    args=(E, L, pot),
                    xtol=_XTOL * r,
                )
        else:
            if self._gamma != 0.0:
                startsign = _rapRperiAxiEq(r, E, L, pot)
                startsign /= numpy.fabs(startsign)
            else:
                startsign = 1.0
            rstart = _rapRperiAxiFindStart(r, E, L, pot, startsign=startsign)
            if rstart == 0.0:
                rperi = 0.0
            else:
                try:
                    rperi = optimize.brentq(
                        _rapRperiAxiEq,
                        rstart,
                        r,
                        (E, L, pot),
                        maxiter=200,
                        xtol=_XTOL * r,
                    )
                except RuntimeError:  # pragma: no cover
                    raise UnboundError("Orbit seems to be unbound")
            rend = _rapRperiAxiFindStart(r, E, L, pot, rap=True, startsign=startsign)
            rap = optimize.brentq(_rapRperiAxiEq, r, rend, (E, L, pot), xtol=_XTOL * r)
        if isinstance(pot, _RelativeEffectivePotential):
            pot.anchor(E, rperi, rap)
        return (rperi, rap)

    def _calc_jr(self, rperi, rap, E, L, fixed_quad, pot=None, **kwargs):
        pot = self._2dpot if pot is None else pot
        if isinstance(pot, _RelativeEffectivePotential) and not fixed_quad:
            # close to the circular orbit: the square-root behaviour of the
            # integrand at the turning points removed by the substitutions
            # r = r_peri + t^2 and r = r_ap - t^2, so that the vectorized
            # Gaussian quadrature converges quickly, and to a relative
            # tolerance, the action being small
            Rmean = 0.5 * (rperi + rap)
            kwargs = {"tol": 0.0, **kwargs}
            return (
                _quadrature(
                    pot,
                    _JrSphericalIntegrandSmall,
                    0.0,
                    numpy.sqrt(Rmean - rperi),
                    args=(E, L, pot, rperi),
                    **kwargs,
                )[0]
                + _quadrature(
                    pot,
                    _JrSphericalIntegrandLarge,
                    0.0,
                    numpy.sqrt(rap - Rmean),
                    args=(E, L, pot, rap),
                    **kwargs,
                )[0]
            ) / numpy.pi
        if fixed_quad:
            return (
                integrate.fixed_quad(
                    _JrSphericalIntegrand,
                    rperi,
                    rap,
                    args=(E, L, pot),
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
                        args=(E, L, pot),
                        **kwargs,
                    )
                )
            )[0] / numpy.pi

    def _calc_or(self, Rmean, rperi, rap, E, L, fixed_quad, pot=None, **kwargs):
        pot = self._2dpot if pot is None else pot
        Tr = 0.0
        if Rmean > rperi and not fixed_quad:
            Tr += numpy.array(
                _quadrature(
                    pot,
                    _TrSphericalIntegrandSmall,
                    0.0,
                    numpy.sqrt(Rmean - rperi),
                    args=(E, L, pot, rperi),
                    **kwargs,
                )
            )[0]
        elif Rmean > rperi and fixed_quad:
            Tr += integrate.fixed_quad(
                _TrSphericalIntegrandSmall,
                0.0,
                numpy.sqrt(Rmean - rperi),
                args=(E, L, pot, rperi),
                n=10,
                **kwargs,
            )[0]
        if Rmean < rap and not fixed_quad:
            Tr += numpy.array(
                _quadrature(
                    pot,
                    _TrSphericalIntegrandLarge,
                    0.0,
                    numpy.sqrt(rap - Rmean),
                    args=(E, L, pot, rap),
                    **kwargs,
                )
            )[0]
        elif Rmean < rap and fixed_quad:
            Tr += integrate.fixed_quad(
                _TrSphericalIntegrandLarge,
                0.0,
                numpy.sqrt(rap - Rmean),
                args=(E, L, pot, rap),
                n=10,
                **kwargs,
            )[0]
        Tr = 2.0 * Tr
        return 2.0 * numpy.pi / Tr

    def _calc_op(self, Or, Rmean, rperi, rap, E, L, fixed_quad, pot=None, **kwargs):
        pot = self._2dpot if pot is None else pot
        # Azimuthal period
        I = 0.0
        if Rmean > rperi and not fixed_quad:
            I += numpy.array(
                _quadrature(
                    pot,
                    _ISphericalIntegrandSmall,
                    0.0,
                    numpy.sqrt(Rmean - rperi),
                    args=(E, L, pot, rperi),
                    **kwargs,
                )
            )[0]
        elif Rmean > rperi and fixed_quad:
            I += integrate.fixed_quad(
                _ISphericalIntegrandSmall,
                0.0,
                numpy.sqrt(Rmean - rperi),
                args=(E, L, pot, rperi),
                n=10,
                **kwargs,
            )[0]
        if Rmean < rap and not fixed_quad:
            I += numpy.array(
                _quadrature(
                    pot,
                    _ISphericalIntegrandLarge,
                    0.0,
                    numpy.sqrt(rap - Rmean),
                    args=(E, L, pot, rap),
                    **kwargs,
                )
            )[0]
        elif Rmean < rap and fixed_quad:
            I += integrate.fixed_quad(
                _ISphericalIntegrandLarge,
                0.0,
                numpy.sqrt(rap - Rmean),
                args=(E, L, pot, rap),
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

    def _calc_angler(
        self, Or, r, Rmean, rperi, rap, E, L, vr, fixed_quad, pot=None, **kwargs
    ):
        pot = self._2dpot if pot is None else pot
        if r < Rmean:
            if r > rperi and not fixed_quad:
                wr = (
                    Or
                    * _quadrature(
                        pot,
                        _TrSphericalIntegrandSmall,
                        0.0,
                        numpy.sqrt(r - rperi),
                        args=(E, L, pot, rperi),
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
                        args=(E, L, pot, rperi),
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
                    * _quadrature(
                        pot,
                        _TrSphericalIntegrandLarge,
                        0.0,
                        numpy.sqrt(rap - r),
                        args=(E, L, pot, rap),
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
                        args=(E, L, pot, rap),
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
        pot=None,
        **kwargs,
    ):
        pot = self._2dpot if pot is None else pot
        psi = self._calc_psi(z, r, L, Lz, vtheta, phi)
        # Calculate dSr/dL
        dpsi = Op / Or * 2.0 * numpy.pi  # this is the full I integral
        if r < Rmean:
            if numpy.sqrt(r - rperi) == 0.0:
                wz = 0.0
            elif not fixed_quad:
                wz = (
                    L
                    * _quadrature(
                        pot,
                        _ISphericalIntegrandSmall,
                        0.0,
                        numpy.sqrt(r - rperi),
                        args=(E, L, pot, rperi),
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
                        args=(E, L, pot, rperi),
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
                    * _quadrature(
                        pot,
                        _ISphericalIntegrandLarge,
                        0.0,
                        numpy.sqrt(rap - r),
                        args=(E, L, pot, rap),
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
                        args=(E, L, pot, rap),
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
    """The J_r integrand, sqrt(2 [E - Phi_eff(r)]); with a relative effective
    potential, E is the energy above the circular orbit's"""
    if isinstance(pot, _RelativeEffectivePotential):
        return numpy.sqrt(2.0 * (E - pot(r)))
    return numpy.sqrt(2.0 * (E - _evaluateplanarPotentials(pot, r)) - L**2.0 / r**2.0)


def _JrSphericalIntegrandSmall(t, E, L, pot, rperi):
    """The J_r integrand in r = r_peri + t^2, relative to the circular orbit;
    the radicand is clipped at zero, where round-off next to the turning
    point would otherwise take it negative (harmless here: nothing divides)"""
    r = rperi + t**2.0
    return 2.0 * t * numpy.sqrt(numpy.fmax(2.0 * (E - pot(r)), 0.0))


def _JrSphericalIntegrandLarge(t, E, L, pot, rap):
    """The J_r integrand in r = r_ap - t^2, relative to the circular orbit"""
    r = rap - t**2.0
    return 2.0 * t * numpy.sqrt(numpy.fmax(2.0 * (E - pot(r)), 0.0))


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
    if isinstance(pot, _RelativeEffectivePotential):
        # E is the energy above the circular orbit's
        return E - pot(R)
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
