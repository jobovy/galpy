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

from ..backend import get_namespace, numpy_island, promote_scalars
from ..potential import IsochronePotential
from ..util import conversion
from .actionAngleInverse import actionAngleInverse


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
            rb = numpy.sqrt(self.b**2.0 + 1.0)
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

    @numpy_island
    def _evaluate(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus.

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

    @numpy_island
    def _xvFreqs(self, jr, jphi, jz, angler, anglephi, anglez, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequencies.

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
        """
        xp = get_namespace(jr, jphi, jz, angler, anglephi, anglez)
        jr, jphi, jz, angler, anglephi, anglez = promote_scalars(
            xp, jr, jphi, jz, angler, anglephi, anglez
        )
        L = jz + xp.abs(jphi)  # total angular momentum
        L2 = L**2.0
        sqrtfourbkL2 = xp.sqrt(L2 + 4.0 * self.b * self.amp)
        H = -2.0 * self.amp**2.0 / (2.0 * jr + L + sqrtfourbkL2) ** 2.0
        # Calculate the frequencies
        omegar = (-2.0 * H) ** 1.5 / self.amp
        omegaz = (1.0 + L / sqrtfourbkL2) / 2.0 * omegar
        # Start on getting the coordinates
        a = -self.amp / 2.0 / H - self.b
        ab = a + self.b
        e = xp.sqrt(1.0 + L2 / (2.0 * H * a**2.0))
        # Solve Kepler's-ish equation eta - (a e/ab) sin(eta) = ar, ar in [0, 2pi)
        angler = (xp.atleast_1d(angler) % (-2.0 * numpy.pi)) % (2.0 * numpy.pi)
        anglephi = xp.atleast_1d(anglephi)
        anglez = xp.atleast_1d(anglez)
        if xp is numpy:
            eta = numpy.empty(len(angler))
            for ii, ar in enumerate(angler):
                try:
                    eta[ii] = optimize.newton(
                        lambda x: x - a * e / ab * numpy.sin(x) - ar,
                        0.0,
                        lambda x: 1 - a * e / ab * numpy.cos(x),
                    )
                except RuntimeError:
                    # Newton-Raphson did not converge, this has to work,
                    # bc 0 <= ra < 2pi the following start x have different signs
                    eta[ii] = optimize.brentq(
                        lambda x: x - a * e / ab * numpy.sin(x) - ar,
                        0.0,
                        2.0 * numpy.pi,
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
        coseta = xp.cos(eta)
        r = a * xp.sqrt((1.0 - e * coseta) * (1.0 - e * coseta + 2.0 * self.b / a))
        vr = xp.sqrt(self.amp / ab) * a * e * xp.sin(eta) / r
        taneta2 = xp.tan(eta / 2.0)
        tan11 = xp.arctan(xp.sqrt((1.0 + e) / (1.0 - e)) * taneta2)
        tan12 = xp.arctan(
            xp.sqrt((a * (1.0 + e) + 2.0 * self.b) / (a * (1.0 - e) + 2.0 * self.b))
            * taneta2
        )
        tan11 = xp.where(tan11 < 0.0, tan11 + numpy.pi, tan11)
        tan12 = xp.where(tan12 < 0.0, tan12 + numpy.pi, tan12)
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

    @numpy_island
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
        H = -2.0 * self.amp**2.0 / (2.0 * jr + L + sqrtfourbkL2) ** 2.0
        # Calculate the frequencies
        omegar = (-2.0 * H) ** 1.5 / self.amp
        omegaz = (1.0 + L / sqrtfourbkL2) / 2.0 * omegar
        return (omegar, xp.sign(jphi) * omegaz, omegaz)
