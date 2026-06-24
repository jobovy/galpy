###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleIsochrone
#
#             Calculate actions-angle coordinates for the Isochrone potential
#
#      methods:
#             __call__: returns (jr,lz,jz)
#             actionsFreqs: returns (jr,lz,jz,Or,Op,Oz)
#             actionsFreqsAngles: returns (jr,lz,jz,Or,Op,Oz,ar,ap,az)
#
###############################################################################
import warnings

import numpy

from ..backend import get_namespace, promote_scalars
from ..potential import IsochronePotential
from ..util import conversion, galpyWarning
from .actionAngle import actionAngle


class actionAngleIsochrone(actionAngle):
    """Action-angle formalism for the isochrone potential, on the Jphi, Jtheta system of Binney & Tremaine (2008)"""

    def __init__(self, *args, **kwargs):
        """
        Initialize an actionAngleIsochrone object.

        Parameters
        ----------
        b : float or Quantity, optional
            Scale parameter of the isochrone parameter.
        ip : IsochronePotential, optional
            Instance of a IsochronePotential.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Specify either b or ip
        - 2013-09-08 - Written - Bovy (IAS)

        """
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))
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
            rb = numpy.sqrt(self.b**2.0 + 1.0)
            self.amp = (self.b + rb) ** 2.0 * rb
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

        Returns
        -------
        tuple
            (jr,lz,jz)

        Notes
        -----
        - 2013-09-08 - Written - Bovy (IAS)
        """
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
        xp = get_namespace(R, vR, vT, z, vz)
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
        if self._c:  # pragma: no cover
            pass
        else:
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            E = self._ip(R, z) + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0
            L = xp.sqrt(L2)
            # Actions
            Jphi = Lz
            Jz = L - xp.abs(Lz)
            Jr = self.amp / xp.sqrt(-2.0 * E) - 0.5 * (
                L + xp.sqrt(L2 + 4.0 * self.amp * self.b)
            )
            return (Jr, Jphi, Jz)

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

        Returns
        -------
        tuple
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)

        Notes
        -----
        - 2013-09-08 - Written - Bovy (IAS)
        """
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
        xp = get_namespace(R, vR, vT, z, vz)
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
        if self._c:  # pragma: no cover
            pass
        else:
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            E = self._ip(R, z) + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0
            L = xp.sqrt(L2)
            # Actions
            Jphi = Lz
            Jz = L - xp.abs(Lz)
            Jr = self.amp / xp.sqrt(-2.0 * E) - 0.5 * (
                L + xp.sqrt(L2 + 4.0 * self.amp * self.b)
            )
            # Frequencies
            Omegar = (-2.0 * E) ** 1.5 / self.amp
            Omegaz = 0.5 * (1.0 + L / xp.sqrt(L2 + 4.0 * self.amp * self.b)) * Omegar
            indx = Lz < 0.0
            Omegaphi = xp.where(indx, -Omegaz, Omegaz)
            return (Jr, Jphi, Jz, Omegar, Omegaphi, Omegaz)

    def _actionsFreqsAngles(self, *args, **kwargs):
        """
        Evaluate the actions, frequencies, and angles (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez).

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
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)

        Notes
        -----
        - 2013-09-08 - Written - Bovy (IAS)
        """
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
        xp = get_namespace(R, vR, vT, z, vz, phi)
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
            phi = numpy.array([phi])
        R, vR, vT, z, vz, phi = promote_scalars(xp, R, vR, vT, z, vz, phi)
        if self._c:  # pragma: no cover
            pass
        else:
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            E = self._ip(R, z) + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0
            L = xp.sqrt(L2)
            # Actions
            Jz = L - xp.abs(Lz)
            Jr = self.amp / xp.sqrt(-2.0 * E) - 0.5 * (
                L + xp.sqrt(L2 + 4.0 * self.amp * self.b)
            )
            # Frequencies
            Omegar = (-2.0 * E) ** 1.5 / self.amp
            Omegaz = 0.5 * (1.0 + L / xp.sqrt(L2 + 4.0 * self.amp * self.b)) * Omegar
            indx = Lz < 0.0
            Omegaphi = xp.where(indx, -Omegaz, Omegaz)
            # Angles
            c = -self.amp / 2.0 / E - self.b
            e2 = 1.0 - L2 / self.amp / c * (1.0 + self.b / c)
            e = xp.sqrt(e2)
            if self.b == 0.0:
                coseta = 1 / e * (1.0 - xp.sqrt(R**2.0 + z**2.0) / c)
            else:
                s = 1.0 + xp.sqrt(1.0 + (R**2.0 + z**2.0) / self.b**2.0)
                coseta = 1 / e * (1.0 - self.b / c * (s - 2.0))
            coseta = xp.clip(coseta, -1.0, 1.0)
            eta = xp.arccos(coseta)
            costheta = z / xp.sqrt(R**2.0 + z**2.0)
            sintheta = R / xp.sqrt(R**2.0 + z**2.0)
            vrindx = (vR * sintheta + vz * costheta) < 0.0
            eta = xp.where(vrindx, 2.0 * numpy.pi - eta, eta)
            angler = eta - e * c / (c + self.b) * xp.sin(eta)
            tan11 = xp.arctan(xp.sqrt((1.0 + e) / (1.0 - e)) * xp.tan(0.5 * eta))
            tan12 = xp.arctan(
                xp.sqrt((1.0 + e + 2.0 * self.b / c) / (1.0 - e + 2.0 * self.b / c))
                * xp.tan(0.5 * eta)
            )
            vzindx = (-vz * sintheta + vR * costheta) > 0.0
            tan11 = xp.where(tan11 < 0.0, tan11 + numpy.pi, tan11)
            tan12 = xp.where(tan12 < 0.0, tan12 + numpy.pi, tan12)
            Lz = xp.where(Lz / L > 1.0, L, Lz)
            Lz = xp.where(Lz / L < -1.0, -L, Lz)
            Jphi = Lz
            sini = xp.sqrt(L**2.0 - Lz**2.0) / L
            tani = xp.sqrt(L**2.0 - Lz**2.0) / Lz
            sinpsi = costheta / sini
            sinpsi = xp.where((sinpsi > 1.0) & xp.isfinite(sinpsi), 1.0, sinpsi)
            sinpsi = xp.where((sinpsi < -1.0) & xp.isfinite(sinpsi), -1.0, sinpsi)
            psi = xp.arcsin(sinpsi)
            psi = xp.where(vzindx, numpy.pi - psi, psi)
            # For non-inclined orbits, we set Omega=0 by convention
            psi = xp.where(~xp.isfinite(psi), phi, psi)
            psi = psi % (2.0 * numpy.pi)
            anglez = (
                psi
                + Omegaz / Omegar * angler
                - tan11
                - 1.0 / xp.sqrt(1.0 + 4 * self.amp * self.b / L2) * tan12
            )
            sinu = z / R / tani
            sinu = xp.where((sinu > 1.0) & xp.isfinite(sinu), 1.0, sinu)
            sinu = xp.where((sinu < -1.0) & xp.isfinite(sinu), -1.0, sinu)
            u = xp.arcsin(sinu)
            u = xp.where(vzindx, numpy.pi - u, u)
            # For non-inclined orbits, we set Omega=0 by convention
            u = xp.where(~xp.isfinite(u), phi, u)
            Omega = phi - u
            anglephi = xp.where(indx, Omega - anglez, Omega + anglez)
            angler = angler % (2.0 * numpy.pi)
            anglephi = anglephi % (2.0 * numpy.pi)
            anglez = anglez % (2.0 * numpy.pi)
            return (Jr, Jphi, Jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez)

    def _EccZmaxRperiRap(self, *args, **kwargs):
        if len(args) == 5:  # R,vR.vT, z, vz pragma: no cover
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
        xp = get_namespace(R, vR, vT, z, vz)
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        R, vR, vT, z, vz = promote_scalars(xp, R, vR, vT, z, vz)
        if self._c:  # pragma: no cover
            pass
        else:
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            E = self._ip(R, z) + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0
            if self.b == 0:
                warnings.warn(
                    "zmax for point-mass (b=0) isochrone potential is only approximate, because it assumes that zmax is attained at rap, which is not necessarily the case",
                    galpyWarning,
                )
                a = -self.amp / 2.0 / E
                me2 = L2 / self.amp / a
                e = xp.sqrt(1.0 - me2)
                rperi = a * (1.0 - e)
                rap = a * (1.0 + e)
            else:
                smin = (
                    0.5
                    * (
                        (2.0 * E - self.amp / self.b)
                        + xp.sqrt(
                            (2.0 * E - self.amp / self.b) ** 2.0
                            + 2.0 * E * (4.0 * self.amp / self.b + L2 / self.b**2.0)
                        )
                    )
                    / E
                )
                smax = 2.0 - self.amp / E / self.b - smin
                rperi = smin * xp.sqrt(1.0 - 2.0 / smin) * self.b
                rap = smax * xp.sqrt(1.0 - 2.0 / smax) * self.b
            return (
                (rap - rperi) / (rap + rperi),
                rap * xp.sqrt(1.0 - Lz**2.0 / L2),
                rperi,
                rap,
            )
