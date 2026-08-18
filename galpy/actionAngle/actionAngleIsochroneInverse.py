###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleIsochroneInverse
#
#             Calculate (x,v) coordinates for the Isochrone potential from
#             given actions-angle coordinates
#
###############################################################################
import copy
import warnings

import numpy
from scipy import optimize

from ..potential import IsochronePotential
from ..util import conversion, galpyWarning
from .actionAngleInverse import actionAngleInverse


def _solve_kepler(k, theta, maxiter, tol=1e-14):
    """Solve the Kepler-like equation eta - k sin(eta) = theta for eta,
    for arrays of k and theta (in [0,2pi]) broadcast against each other.
    Because the left-hand side increases monotonically in eta, the root is
    always bracketed by theta and pi (eta >= theta before apocenter and
    eta <= theta after it), so we can safeguard Newton-Raphson with
    bisection: the Newton-Raphson iteration converges quadratically except
    near eta = 0 for nearly-radial tori (where k -> 1 and the derivative
    vanishes), where the bisection fallback takes over. Only the entries
    that have not converged yet are iterated."""
    k, theta = numpy.broadcast_arrays(k, theta)
    shape = theta.shape
    k = numpy.ascontiguousarray(k, dtype="float").flatten()
    theta = numpy.ascontiguousarray(theta, dtype="float").flatten()
    eta = copy.copy(theta)
    brlo = numpy.minimum(theta, numpy.pi)
    brhi = numpy.maximum(theta, numpy.pi)
    # Indices of the points that have not converged yet
    unconv = numpy.arange(theta.size)
    cntr = 0
    while unconv.size:
        teta, tk = eta[unconv], k[unconv]
        tbrlo, tbrhi = brlo[unconv], brhi[unconv]
        f = teta - tk * numpy.sin(teta) - theta[unconv]
        tbrlo = numpy.where(f <= 0.0, teta, tbrlo)
        tbrhi = numpy.where(f >= 0.0, teta, tbrhi)
        # The derivative only vanishes for a radial orbit at pericenter,
        # where the bisection fallback below takes over anyway
        deriv = 1.0 - tk * numpy.cos(teta)
        newteta = teta - f / numpy.where(deriv < 1e-300, 1e-300, deriv)
        # Fall back onto bisection when Newton-Raphson leaves the bracket
        bisect = (newteta <= tbrlo) + (newteta >= tbrhi)
        newteta[bisect] = 0.5 * (tbrlo[bisect] + tbrhi[bisect])
        conv = numpy.fabs(newteta - teta) < tol
        eta[unconv] = newteta
        brlo[unconv] = tbrlo
        brhi[unconv] = tbrhi
        unconv = unconv[True ^ conv]
        cntr += 1
        if cntr > maxiter:
            warnings.warn(
                "Solving the Kepler-like equation for the auxiliary eccentric anomaly did not converge in {} iterations".format(
                    maxiter
                ),
                galpyWarning,
            )
            break
    return eta.reshape(shape)


# Maximum number of iterations for the (safeguarded) solve of the
# Kepler-like equation; bisection alone converges in about 50
_KEPLER_MAXITER = 100


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
        jr = numpy.atleast_1d(jr)
        L = jz + numpy.fabs(jphi)  # total angular momentum
        L2 = L**2.0
        sqrtfourbkL2 = numpy.sqrt(L2 + 4.0 * self.b * self.amp)
        H = -2.0 * self.amp**2.0 / (2.0 * jr + L + sqrtfourbkL2) ** 2.0
        # Calculate the frequencies
        omegar = (-2.0 * H) ** 1.5 / self.amp
        omegaz = (1.0 + L / sqrtfourbkL2) / 2.0 * omegar
        # Start on getting the coordinates
        a = -self.amp / 2.0 / H - self.b
        ab = a + self.b
        e = numpy.sqrt(1.0 + L2 / (2.0 * H * a**2.0))
        # Solve Kepler's-ish equation; ar must be between 0 and 2pi
        angler = (numpy.atleast_1d(angler) % (-2.0 * numpy.pi)) % (2.0 * numpy.pi)
        anglephi = numpy.atleast_1d(anglephi)
        anglez = numpy.atleast_1d(anglez)
        aeoverab = a * e / ab
        # Solved for all points at once with Newton-Raphson safeguarded by
        # bisection (sometimes jr is an array, sometimes it is a single
        # torus evaluated at many angles)
        eta = _solve_kepler(
            aeoverab if len(jr) > 1 else aeoverab[0], angler, _KEPLER_MAXITER
        )
        coseta = numpy.cos(eta)
        r = a * numpy.sqrt((1.0 - e * coseta) * (1.0 - e * coseta + 2.0 * self.b / a))
        vr = numpy.sqrt(self.amp / ab) * a * e * numpy.sin(eta) / r
        taneta2 = numpy.tan(eta / 2.0)
        tan11 = numpy.arctan(numpy.sqrt((1.0 + e) / (1.0 - e)) * taneta2)
        tan12 = numpy.arctan(
            numpy.sqrt((a * (1.0 + e) + 2.0 * self.b) / (a * (1.0 - e) + 2.0 * self.b))
            * taneta2
        )
        tan11[tan11 < 0.0] += numpy.pi
        tan12[tan12 < 0.0] += numpy.pi
        Lambdaeta = tan11 + L / sqrtfourbkL2 * tan12
        psi = anglez - omegaz / omegar * angler + Lambdaeta
        lowerl = numpy.sqrt(1.0 - jphi**2.0 / L2)
        costheta = numpy.sin(psi) * lowerl
        sintheta = numpy.sqrt(1.0 - costheta**2.0)
        vtheta = -L * lowerl * numpy.cos(psi) / sintheta / r
        R = r * sintheta
        z = r * costheta
        vR = vr * sintheta + vtheta * costheta
        vz = vr * costheta - vtheta * sintheta
        sinu = costheta / sintheta * jphi / L / lowerl
        u = numpy.arcsin(sinu)
        u[vtheta > 0.0] = numpy.pi - u[vtheta > 0.0]
        phi = anglephi - numpy.sign(jphi) * anglez + u
        # For non-inclined orbits, phi == psi
        phi[True ^ numpy.isfinite(phi)] = psi[True ^ numpy.isfinite(phi)]
        phi = phi % (2.0 * numpy.pi)
        phi[phi < 0.0] += 2.0 * numpy.pi
        return (R, vR, jphi / R, z, vz, phi, omegar, numpy.sign(jphi) * omegaz, omegaz)

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
        L = jz + numpy.fabs(jphi)  # total angular momentum
        sqrtfourbkL2 = numpy.sqrt(L**2.0 + 4.0 * self.b * self.amp)
        H = -2.0 * self.amp**2.0 / (2.0 * jr + L + sqrtfourbkL2) ** 2.0
        # Calculate the frequencies
        omegar = (-2.0 * H) ** 1.5 / self.amp
        omegaz = (1.0 + L / sqrtfourbkL2) / 2.0 * omegar
        return (omegar, numpy.sign(jphi) * omegaz, omegaz)
