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
import copy
import warnings
import numpy
from .actionAngle import actionAngle
from ..potential import IsochronePotential
from ..util import galpyWarning, conversion


class actionAngleIsochrone(actionAngle):
    """Action-angle formalism for the isochrone potential, on the Jphi, Jtheta system of Binney & Tremaine (2008)"""

    def __init__(self, *args, **kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleIsochrone object
        INPUT:
           Either:

              b= scale parameter of the isochrone parameter (can be Quantity)

              ip= instance of a IsochronePotential

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:

           instance

        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
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
        NAME:
           __call__ (_evaluate)
        PURPOSE:
           evaluate the actions (jr,lz,jz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        OUTPUT:
           (jr,lz,jz)
        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
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
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        if self._c:  # pragma: no cover
            pass
        else:
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            E = self._ip(R, z) + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0
            L = numpy.sqrt(L2)
            # Actions
            Jphi = Lz
            Jz = L - numpy.fabs(Lz)
            Jr = self.amp / numpy.sqrt(-2.0 * E) - 0.5 * (
                L + numpy.sqrt(L2 + 4.0 * self.amp * self.b)
            )
            return (Jr, Jphi, Jz)

    def _actionsFreqs(self, *args, **kwargs):
        """
        NAME:
           actionsFreqs (_actionsFreqs)
        PURPOSE:
           evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
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
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
        if self._c:  # pragma: no cover
            pass
        else:
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            E = self._ip(R, z) + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0
            L = numpy.sqrt(L2)
            # Actions
            Jphi = Lz
            Jz = L - numpy.fabs(Lz)
            Jr = self.amp / numpy.sqrt(-2.0 * E) - 0.5 * (
                L + numpy.sqrt(L2 + 4.0 * self.amp * self.b)
            )
            # Frequencies
            Omegar = (-2.0 * E) ** 1.5 / self.amp
            Omegaz = 0.5 * (1.0 + L / numpy.sqrt(L2 + 4.0 * self.amp * self.b)) * Omegar
            Omegaphi = copy.copy(Omegaz)
            indx = Lz < 0.0
            Omegaphi[indx] *= -1.0
            return (Jr, Jphi, Jz, Omegar, Omegaphi, Omegaz)

    def _actionsFreqsAngles(self, *args, **kwargs):
        """
        NAME:
           actionsFreqsAngles (_actionsFreqsAngles)
        PURPOSE:
           evaluate the actions, frequencies, and angles (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
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
            Lz = R * vT
            Lx = -z * vT
            Ly = z * vR - R * vz
            L2 = Lx * Lx + Ly * Ly + Lz * Lz
            E = self._ip(R, z) + vR**2.0 / 2.0 + vT**2.0 / 2.0 + vz**2.0 / 2.0
            L = numpy.sqrt(L2)
            # Actions
            Jphi = Lz
            Jz = L - numpy.fabs(Lz)
            Jr = self.amp / numpy.sqrt(-2.0 * E) - 0.5 * (
                L + numpy.sqrt(L2 + 4.0 * self.amp * self.b)
            )
            # Frequencies
            Omegar = (-2.0 * E) ** 1.5 / self.amp
            Omegaz = 0.5 * (1.0 + L / numpy.sqrt(L2 + 4.0 * self.amp * self.b)) * Omegar
            Omegaphi = copy.copy(Omegaz)
            indx = Lz < 0.0
            Omegaphi[indx] *= -1.0
            # Angles
            c = -self.amp / 2.0 / E - self.b
            e2 = 1.0 - L2 / self.amp / c * (1.0 + self.b / c)
            e = numpy.sqrt(e2)
            if self.b == 0.0:
                coseta = 1 / e * (1.0 - numpy.sqrt(R**2.0 + z**2.0) / c)
            else:
                s = 1.0 + numpy.sqrt(1.0 + (R**2.0 + z**2.0) / self.b**2.0)
                coseta = 1 / e * (1.0 - self.b / c * (s - 2.0))
            pindx = coseta > 1.0
            coseta[pindx] = 1.0
            pindx = coseta < -1.0
            coseta[pindx] = -1.0
            eta = numpy.arccos(coseta)
            costheta = z / numpy.sqrt(R**2.0 + z**2.0)
            sintheta = R / numpy.sqrt(R**2.0 + z**2.0)
            vrindx = (vR * sintheta + vz * costheta) < 0.0
            eta[vrindx] = 2.0 * numpy.pi - eta[vrindx]
            angler = eta - e * c / (c + self.b) * numpy.sin(eta)
            tan11 = numpy.arctan(
                numpy.sqrt((1.0 + e) / (1.0 - e)) * numpy.tan(0.5 * eta)
            )
            tan12 = numpy.arctan(
                numpy.sqrt((1.0 + e + 2.0 * self.b / c) / (1.0 - e + 2.0 * self.b / c))
                * numpy.tan(0.5 * eta)
            )
            vzindx = (-vz * sintheta + vR * costheta) > 0.0
            tan11[tan11 < 0.0] += numpy.pi
            tan12[tan12 < 0.0] += numpy.pi
            pindx = Lz / L > 1.0
            Lz[pindx] = L[pindx]
            pindx = Lz / L < -1.0
            Lz[pindx] = -L[pindx]
            sini = numpy.sqrt(L**2.0 - Lz**2.0) / L
            tani = numpy.sqrt(L**2.0 - Lz**2.0) / Lz
            sinpsi = costheta / sini
            pindx = (sinpsi > 1.0) * numpy.isfinite(sinpsi)
            sinpsi[pindx] = 1.0
            pindx = (sinpsi < -1.0) * numpy.isfinite(sinpsi)
            sinpsi[pindx] = -1.0
            psi = numpy.arcsin(sinpsi)
            psi[vzindx] = numpy.pi - psi[vzindx]
            # For non-inclined orbits, we set Omega=0 by convention
            psi[True ^ numpy.isfinite(psi)] = phi[True ^ numpy.isfinite(psi)]
            psi = psi % (2.0 * numpy.pi)
            anglez = (
                psi
                + Omegaz / Omegar * angler
                - tan11
                - 1.0 / numpy.sqrt(1.0 + 4 * self.amp * self.b / L2) * tan12
            )
            sinu = z / R / tani
            pindx = (sinu > 1.0) * numpy.isfinite(sinu)
            sinu[pindx] = 1.0
            pindx = (sinu < -1.0) * numpy.isfinite(sinu)
            sinu[pindx] = -1.0
            u = numpy.arcsin(sinu)
            u[vzindx] = numpy.pi - u[vzindx]
            # For non-inclined orbits, we set Omega=0 by convention
            u[True ^ numpy.isfinite(u)] = phi[True ^ numpy.isfinite(u)]
            Omega = phi - u
            anglephi = Omega
            anglephi[indx] -= anglez[indx]
            anglephi[True ^ indx] += anglez[True ^ indx]
            angler = angler % (2.0 * numpy.pi)
            anglephi = anglephi % (2.0 * numpy.pi)
            anglez = anglez % (2.0 * numpy.pi)
            return (Jr, Jphi, Jz, Omegar, Omegaphi, Omegaz, angler, anglephi, anglez)

    def _EccZmaxRperiRap(self, *args, **kwargs):
        """
        NAME:
           _EccZmaxRperiRap
        PURPOSE:
           evaluate the eccentricity, maximum height above the plane, peri- and apocenter for an isochrone potential
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        OUTPUT:
           (e,zmax,rperi,rap)
        HISTORY:
           2017-12-22 - Written - Bovy (UofT)
        """
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
        if isinstance(R, float):
            R = numpy.array([R])
            vR = numpy.array([vR])
            vT = numpy.array([vT])
            z = numpy.array([z])
            vz = numpy.array([vz])
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
                e = numpy.sqrt(1.0 - me2)
                rperi = a * (1.0 - e)
                rap = a * (1.0 + e)
            else:
                smin = (
                    0.5
                    * (
                        (2.0 * E - self.amp / self.b)
                        + numpy.sqrt(
                            (2.0 * E - self.amp / self.b) ** 2.0
                            + 2.0 * E * (4.0 * self.amp / self.b + L2 / self.b**2.0)
                        )
                    )
                    / E
                )
                smax = 2.0 - self.amp / E / self.b - smin
                rperi = smin * numpy.sqrt(1.0 - 2.0 / smin) * self.b
                rap = smax * numpy.sqrt(1.0 - 2.0 / smax) * self.b
            return (
                (rap - rperi) / (rap + rperi),
                rap * numpy.sqrt(1.0 - Lz**2.0 / L2),
                rperi,
                rap,
            )


class _actionAngleIsochroneHelper(object):
    """Simplified version of the actionAngleIsochrone transformations, for use in actionAngleSphericalInverse"""

    def __init__(self, *args, **kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an _actionAngleIsochroneHelper object

        INPUT:

           ip= instance of a IsochronePotential

        OUTPUT:

           instance

        HISTORY:

           2017-11-30 - Written - Bovy (UofT)

        """
        if not "ip" in kwargs:  # pragma: no cover
            raise IOError("Must specify ip= for _actionAngleIsochroneHelper")
        else:
            ip = kwargs["ip"]
            if not isinstance(ip, IsochronePotential):  # pragma: no cover
                raise IOError(
                    "'Provided ip= does not appear to be an instance of an IsochronePotential"
                )
            # Check the units
            self.b = ip.b
            self.amp = ip._amp
        self._ip = ip
        return None

    def angler(self, r, vr2, L, reuse=False, vrneg=False):
        """
        NAME:
           angler
        PURPOSE:
           calculate the radial angle
        INPUT:
           r - radius
           vr2 - radial velocity squared
           L - angular momentum
           vrneg= (False) True if vr is negative
           reuse= (False) if True, re-use all relevant quantities for computing the radial angle that were computed prviously as part of danglerdr_constant_L)
        OUTPUT:
           radial angle
        HISTORY:
           2017-11-30 - Written - Bovy (UofT)
        """
        if reuse:
            return (
                self._eta - self._e * self._c / (self._c + self.b) * self._sineta
            ) % (2.0 * numpy.pi)
        E = self._ip(r, 0.0) + vr2 / 2.0 + L**2.0 / 2.0 / r**2.0
        # if E > 0.: return -1.
        c = -self.amp / 2.0 / E - self.b
        e2 = 1.0 - L * L / self.amp / c * (1.0 + self.b / c)
        e = numpy.sqrt(e2)
        if isinstance(self.b, numpy.ndarray):
            s = 1.0 + numpy.sqrt(1.0 + r * r / self.b**2.0)
            coseta = 1 / e * (1.0 - self.b / c * (s - 2.0))
            pindx = self.b == 0.0
            coseta[pindx] = 1 / e[pindx] * (1.0 - r[pindx] / c[pindx])
        else:
            if self.b == 0.0:
                coseta = 1 / e * (1.0 - r / c)
            else:
                s = 1.0 + numpy.sqrt(1.0 + r * r / self.b**2.0)
                coseta = 1 / e * (1.0 - self.b / c * (s - 2.0))
        pindx = coseta > 1.0
        coseta[pindx] = 1.0
        pindx = coseta < -1.0
        coseta[pindx] = -1.0
        eta = numpy.arccos(coseta)
        if vrneg:
            eta = 2.0 * numpy.pi - eta
        angler = (eta - e * c / (c + self.b) * numpy.sin(eta)) % (2.0 * numpy.pi)
        angler[E > 0.0] = -1.0
        return angler

    def danglerdr_constant_L(self, r, vr2, L, dEdr, vrneg=False):
        """Function used in actionAngleSphericalInverse when finding r at which angler has a particular value on the isochrone torus"""
        E = self._ip(r, 0.0) + vr2 / 2.0 + L**2.0 / 2.0 / r**2.0
        L2 = L**2.0
        self._c = -self.amp / 2.0 / E - self.b
        L2overampc = L2 / self.amp / self._c
        e2 = 1.0 - L2overampc * (1.0 + self.b / self._c)
        self._e = numpy.sqrt(e2)
        if isinstance(self.b, numpy.ndarray):
            s = 1.0 + numpy.sqrt(1.0 + r * r / self.b**2.0)
            coseta = 1 / self._e * (1.0 - self.b / self._c * (s - 2.0))
            pindx = self.b == 0.0
            coseta[pindx] = 1 / self._e[pindx] * (1.0 - r[pindx] / self._c[pindx])
        else:
            if self.b == 0.0:
                coseta = 1 / self._e * (1.0 - r / self._c)
            else:
                s = 1.0 + numpy.sqrt(1.0 + r * r / self.b**2.0)
                coseta = 1 / self._e * (1.0 - self.b / self._c * (s - 2.0))
        pindx = coseta > 1.0
        coseta[pindx] = 1.0
        pindx = coseta < -1.0
        coseta[pindx] = -1.0
        self._eta = numpy.arccos(coseta)
        if vrneg:
            self._eta = 2.0 * numpy.pi - self._eta
        self._sineta = numpy.sin(self._eta)
        L2overampc *= (1.0 + 2.0 * self.b / self._c) / (
            2.0 * self._e
        )  # from now on need L2/(2GM c e)
        dcdr = self.amp / 2.0 / E**2.0 * dEdr
        dsdrtimesb = r / numpy.sqrt(r**2.0 + self.b**2.0)
        detadr = (dsdrtimesb + (coseta * (self._e + L2overampc) - 1.0) * dcdr) / (
            self._e * self._c * self._sineta
        )
        return (
            detadr * (1.0 - self._e * self._c * coseta / (self._c + self.b))
            - self._sineta
            / (self._c + self.b)
            * (self._e * self.b / (self._c + self.b) + L2overampc)
            * dcdr
        )

    def Jr(self, E, L):
        return self.amp / numpy.sqrt(-2.0 * E) - 0.5 * (
            L + numpy.sqrt((L * L + 4.0 * self.amp * self.b))
        )

    def Or(self, E):
        return (-2.0 * E) ** 1.5 / self.amp

    def rperirap(self, E, L2):
        if self.b == 0:
            a = -self.amp / 2.0 / E
            me2 = L2 / self.amp / a
            e = numpy.sqrt(1.0 - me2)
            rperi = a * (1.0 - e)
            rap = a * (1.0 + e)
        else:
            smin = (
                0.5
                * (
                    (2.0 * E - self.amp / self.b)
                    + numpy.sqrt(
                        (2.0 * E - self.amp / self.b) ** 2.0
                        + 2.0 * E * (4.0 * self.amp / self.b + L2 / self.b**2.0)
                    )
                )
                / E
            )
            smax = 2.0 - self.amp / E / self.b - smin
            rperi = smin * numpy.sqrt(1.0 - 2.0 / smin) * self.b
            rap = smax * numpy.sqrt(1.0 - 2.0 / smax) * self.b
        return (rperi, rap)

    def drdEL_constant_angler(self, r, vr2, E, L, dEdr, dEdL, vrneg=False):
        """Function used in actionAngleSphericalInverse to determine dEA/dE and dEA/dL: derivative of the radius r wrt E and L necessary to have constant angler"""
        L2 = L**2.0
        c = -self.amp / 2.0 / E - self.b
        e2 = 1.0 - L2 / self.amp / c * (1.0 + self.b / c)
        e = numpy.sqrt(e2)
        if isinstance(self.b, numpy.ndarray):
            s = 1.0 + numpy.sqrt(1.0 + r * r / self.b**2.0)
            coseta = 1 / e * (1.0 - self.b / c * (s - 2.0))
            pindx = self.b == 0.0
            coseta[pindx] = 1 / e[pindx] * (1.0 - r[pindx] / c[pindx])
        else:
            if self.b == 0.0:
                coseta = 1 / e * (1.0 - r / c)
            else:
                s = 1.0 + numpy.sqrt(1.0 + r * r / self.b**2.0)
                coseta = 1 / e * (1.0 - self.b / c * (s - 2.0))
        pindx = coseta > 1.0
        coseta[pindx] = 1.0
        pindx = coseta < -1.0
        coseta[pindx] = -1.0
        eta = numpy.arccos(coseta)
        if vrneg:
            eta = 2.0 * numpy.pi - eta
        sineta = numpy.sin(eta)
        bcmecce = self.b + c - e * c * coseta
        c2e2ob = c**2.0 * sineta**2.0 / self.b
        dcdLfac = (1.0 - e * coseta) / self.b + e2 * c2e2ob / bcmecce * (
            1.0 / c - 1.0 / (self.b + c)
        )
        dcdLoverdEdL = self.amp / 2.0 / E**2.0
        dcdLoverdrdL = dcdLoverdEdL * dEdr
        dedLfac = -c * coseta / e / self.b + c2e2ob / bcmecce
        L2o2GMc2etc = L2 / 2.0 / self.amp / c**2.0 * (1.0 + 2.0 * self.b / c)
        numfordrdE = (
            dcdLfac * self.amp / 2.0 / E** 2.0
            + dedLfac * L2 / 4.0 / c** 2.0 / E** 2 * (1.0 + 2.0 * self.b / c)
        )
        return (
            numfordrdE / (r / self.b**2.0 / (s - 1.0) - numfordrdE * dEdr),
            (
                -dedLfac
                * (
                    L / self.amp / c * (1.0 + self.b / c)
                    - L2o2GMc2etc * dcdLoverdEdL * dEdL
                )
                + dcdLfac * dcdLoverdEdL * dEdL
            )
            / (
                r / self.b**2.0 / (s - 1.0)
                - dcdLfac * dcdLoverdrdL
                - dedLfac * L2o2GMc2etc * dcdLoverdrdL
            ),
        )
