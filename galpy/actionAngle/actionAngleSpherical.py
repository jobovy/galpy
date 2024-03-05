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

from ..potential import _dim, epifreq, omegac, vcirc
from ..potential.planarPotential import _evaluateplanarPotentials
from ..potential.Potential import _evaluatePotentials
from ..potential.Potential import flatten as flatten_potential
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
        pot : Potential or list thereof
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
        self._pot = flatten_potential(kwargs["pot"])
        # Also store a 'planar' (2D) version of the potential, only potential
        # used in this class
        if _dim(self._pot) == 2:
            self._2dpot = self._pot
        elif isinstance(self._pot, list):
            self._2dpot = [p.toPlanar() for p in self._pot]
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
                wr = numpy.pi
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
            if not fixed_quad:
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
            if not fixed_quad:
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
    pot : Potential object or list thereof
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
