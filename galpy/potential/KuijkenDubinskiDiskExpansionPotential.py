###############################################################################
#   KuijkenDubinskiDiskExpansionPotential.py: Base class for disk+halo
#   potentials using the Kuijken & Dubinski (1995) technique
###############################################################################
import copy

import numpy
import scipy
from packaging.version import parse as parse_version

_SCIPY_VERSION = parse_version(scipy.__version__)
if _SCIPY_VERSION < parse_version("0.10"):  # pragma: no cover
    from scipy.maxentropy import logsumexp
elif _SCIPY_VERSION < parse_version("0.19"):  # pragma: no cover
    from scipy.misc import logsumexp
else:
    from scipy.special import logsumexp

from scipy import integrate

from .Potential import Potential


class KuijkenDubinskiDiskExpansionPotential(Potential):
    """Base class for disk+halo potentials using the Kuijken & Dubinski (1995)
    technique. This class contains all shared logic for decomposing a potential
    as Phi = Phi_ME + 4*pi*G * sum_i Sigma_i(r) * H_i(z), where Phi_ME is
    solved via a multipole/basis-function expansion.

    Subclasses (DiskSCFPotential, DiskMultipoleExpansionPotential) only differ
    in how they create the expansion sub-potential (self._me).
    """

    def __init__(
        self,
        amp=1.0,
        dens=lambda R, z: 13.5 * numpy.exp(-3.0 * R) * numpy.exp(-27.0 * numpy.fabs(z)),
        Sigma={"type": "exp", "h": 1.0 / 3.0, "amp": 1.0},
        hz={"type": "exp", "h": 1.0 / 27.0},
        Sigma_amp=None,
        dSigmadR=None,
        d2SigmadR2=None,
        Hz=None,
        dHzdz=None,
        ro=None,
        vo=None,
    ):
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=None)
        # Parse and store given functions
        self.isNonAxi = dens.__code__.co_argcount == 3
        self._parse_Sigma(Sigma_amp, Sigma, dSigmadR, d2SigmadR2)
        self._parse_hz(hz, Hz, dHzdz)
        if self.isNonAxi:
            self._inputdens = dens
        else:
            self._inputdens = lambda R, z, phi: dens(R, z)
        # Compute phiME density function
        if not self.isNonAxi:
            self._phiME_dens_func = lambda R, z: phiME_dens(
                R,
                z,
                0.0,
                self._inputdens,
                self._Sigma,
                self._dSigmadR,
                self._d2SigmadR2,
                self._hz,
                self._Hz,
                self._dHzdz,
                self._Sigma_amp,
            )
        else:
            self._phiME_dens_func = lambda R, z, phi: phiME_dens(
                R,
                z,
                phi,
                self._inputdens,
                self._Sigma,
                self._dSigmadR,
                self._d2SigmadR2,
                self._hz,
                self._Hz,
                self._dHzdz,
                self._Sigma_amp,
            )

    def _finish_init(self, normalize):
        """Called by subclasses after setting self._me."""
        if self._Sigma_dict is not None and self._hz_dict is not None:
            self.hasC = True
            self.hasC_dens = True
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            self.normalize(normalize)

    def _parse_Sigma(self, Sigma_amp, Sigma, dSigmadR, d2SigmadR2):
        if isinstance(Sigma, dict):
            Sigma = [Sigma]
        try:
            nsigma = len(Sigma)
        except TypeError:
            Sigma_amp = [Sigma_amp]
            Sigma = [Sigma]
            dSigmadR = [dSigmadR]
            d2SigmadR2 = [d2SigmadR2]
            nsigma = 1
        self._nsigma = nsigma
        self._Sigma_amp = Sigma_amp
        self._Sigma = Sigma
        self._dSigmadR = dSigmadR
        self._d2SigmadR2 = d2SigmadR2
        if isinstance(Sigma[0], dict):
            self._Sigma_dict = copy.copy(Sigma)
            self._parse_Sigma_dict()
        else:
            self._Sigma_dict = None
        return None

    def _parse_Sigma_dict(self):
        Sigma_amp, Sigma, dSigmadR, d2SigmadR2 = [], [], [], []
        for ii in range(self._nsigma):
            ta, ts, tds, td2s = self._parse_Sigma_dict_indiv(self._Sigma[ii])
            Sigma_amp.append(ta)
            Sigma.append(ts)
            dSigmadR.append(tds)
            d2SigmadR2.append(td2s)
        self._Sigma_amp = Sigma_amp
        self._Sigma = Sigma
        self._dSigmadR = dSigmadR
        self._d2SigmadR2 = d2SigmadR2
        return None

    def _parse_Sigma_dict_indiv(self, Sigma):
        stype = Sigma.get("type", "exp")
        if stype == "exp" and not "Rhole" in Sigma:
            rd = Sigma.get("h", 1.0 / 3.0)
            ta = Sigma.get("amp", 1.0)
            ts = lambda R, trd=rd: numpy.exp(-R / trd)
            tds = lambda R, trd=rd: -numpy.exp(-R / trd) / trd
            td2s = lambda R, trd=rd: numpy.exp(-R / trd) / trd**2.0
        elif stype == "expwhole" or (stype == "exp" and "Rhole" in Sigma):
            rd = Sigma.get("h", 1.0 / 3.0)
            rm = Sigma.get("Rhole", 0.5)
            ta = Sigma.get("amp", 1.0)
            ts = lambda R, trd=rd, trm=rm: numpy.exp(-trm / R - R / trd)
            tds = lambda R, trd=rd, trm=rm: (trm / R**2.0 - 1.0 / trd) * numpy.exp(
                -trm / R - R / trd
            )
            td2s = lambda R, trd=rd, trm=rm: (
                (trm / R**2.0 - 1.0 / trd) ** 2.0 - 2.0 * trm / R**3.0
            ) * numpy.exp(-trm / R - R / trd)
        return (ta, ts, tds, td2s)

    def _parse_hz(self, hz, Hz, dHzdz):
        if isinstance(hz, dict):
            hz = [hz]
        try:
            nhz = len(hz)
        except TypeError:
            hz = [hz]
            Hz = [Hz]
            dHzdz = [dHzdz]
            nhz = 1
        if nhz != self._nsigma and nhz != 1:
            raise ValueError(
                "Number of hz functions needs to be equal to the number of Sigma functions or to 1"
            )
        if nhz == 1 and self._nsigma > 1:
            hz = [hz[0] for ii in range(self._nsigma)]
            if not isinstance(hz[0], dict):
                Hz = [Hz[0] for ii in range(self._nsigma)]
                dHzdz = [dHzdz[0] for ii in range(self._nsigma)]
        self._Hz = Hz
        self._hz = hz
        self._dHzdz = dHzdz
        self._nhz = len(self._hz)
        if isinstance(hz[0], dict):
            self._hz_dict = copy.copy(hz)
            self._parse_hz_dict()
        else:
            self._hz_dict = None
        return None

    def _parse_hz_dict(self):
        hz, Hz, dHzdz = [], [], []
        for ii in range(self._nhz):
            th, tH, tdH = self._parse_hz_dict_indiv(self._hz[ii])
            hz.append(th)
            Hz.append(tH)
            dHzdz.append(tdH)
        self._hz = hz
        self._Hz = Hz
        self._dHzdz = dHzdz
        return None

    def _parse_hz_dict_indiv(self, hz):
        htype = hz.get("type", "exp")
        if htype == "exp":
            zd = hz.get("h", 0.0375)
            th = lambda z, tzd=zd: 1.0 / 2.0 / tzd * numpy.exp(-numpy.fabs(z) / tzd)
            tH = (
                lambda z, tzd=zd: (
                    numpy.exp(-numpy.fabs(z) / tzd) - 1.0 + numpy.fabs(z) / tzd
                )
                * tzd
                / 2.0
            )
            tdH = (
                lambda z, tzd=zd: 0.5
                * numpy.sign(z)
                * (1.0 - numpy.exp(-numpy.fabs(z) / tzd))
            )
        elif htype == "sech2":
            zd = hz.get("h", 0.0375)
            # th/tH written so as to avoid overflow in cosh
            th = (
                lambda z, tzd=zd: numpy.exp(
                    -logsumexp(
                        numpy.array(
                            [z / tzd, -z / tzd, numpy.log(2.0) * numpy.ones_like(z)]
                        ),
                        axis=0,
                    )
                )
                / tzd
            )
            tH = lambda z, tzd=zd: tzd * (
                logsumexp(numpy.array([z / 2.0 / tzd, -z / 2.0 / tzd]), axis=0)
                - numpy.log(2.0)
            )
            tdH = lambda z, tzd=zd: numpy.tanh(z / 2.0 / tzd) / 2.0
        return (th, tH, tdH)

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        out = self._me(R, z, phi=phi, use_physical=False)
        for a, s, H in zip(self._Sigma_amp, self._Sigma, self._Hz):
            out += 4.0 * numpy.pi * a * s(r) * H(z)
        return out

    def _Rforce(self, R, z, phi=0, t=0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        out = self._me.Rforce(R, z, phi=phi, use_physical=False)
        for a, ds, H in zip(self._Sigma_amp, self._dSigmadR, self._Hz):
            out -= 4.0 * numpy.pi * a * ds(r) * H(z) * R / r
        return out

    def _zforce(self, R, z, phi=0, t=0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        out = self._me.zforce(R, z, phi=phi, use_physical=False)
        for a, s, ds, H, dH in zip(
            self._Sigma_amp, self._Sigma, self._dSigmadR, self._Hz, self._dHzdz
        ):
            out -= 4.0 * numpy.pi * a * (ds(r) * H(z) * z / r + s(r) * dH(z))
        return out

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        return self._me.phitorque(R, z, phi=phi, use_physical=False)

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        out = self._me.R2deriv(R, z, phi=phi, use_physical=False)
        for a, ds, d2s, H in zip(
            self._Sigma_amp, self._dSigmadR, self._d2SigmadR2, self._Hz
        ):
            out += (
                4.0
                * numpy.pi
                * a
                * H(z)
                / r**2.0
                * (d2s(r) * R**2.0 + z**2.0 / r * ds(r))
            )
        return out

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        out = self._me.z2deriv(R, z, phi=phi, use_physical=False)
        for a, s, ds, d2s, h, H, dH in zip(
            self._Sigma_amp,
            self._Sigma,
            self._dSigmadR,
            self._d2SigmadR2,
            self._hz,
            self._Hz,
            self._dHzdz,
        ):
            out += (
                4.0
                * numpy.pi
                * a
                * (
                    H(z) / r**2.0 * (d2s(r) * z**2.0 + ds(r) * R**2.0 / r)
                    + 2.0 * ds(r) * dH(z) * z / r
                    + s(r) * h(z)
                )
            )
        return out

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        out = self._me.Rzderiv(R, z, phi=phi, use_physical=False)
        for a, ds, d2s, H, dH in zip(
            self._Sigma_amp, self._dSigmadR, self._d2SigmadR2, self._Hz, self._dHzdz
        ):
            out += (
                4.0
                * numpy.pi
                * a
                * (H(z) * R * z / r**2.0 * (d2s(r) - ds(r) / r) + ds(r) * dH(z) * R / r)
            )
        return out

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        return self._me.phi2deriv(R, z, phi=phi, use_physical=False)

    def _dens(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        out = self._me.dens(R, z, phi=phi, use_physical=False)
        for a, s, ds, d2s, h, H, dH in zip(
            self._Sigma_amp,
            self._Sigma,
            self._dSigmadR,
            self._d2SigmadR2,
            self._hz,
            self._Hz,
            self._dHzdz,
        ):
            out += a * (
                s(r) * h(z) + d2s(r) * H(z) + 2.0 / r * ds(r) * (H(z) + z * dH(z))
            )
        return out

    def _mass(self, R, z=None, t=0.0):
        if not z is None:  # pragma: no cover
            raise AttributeError  # Hack to fall back to general
        out = self._me.mass(R, z=None, use_physical=False)
        r = R

        def _integrand(theta):
            # ~ rforce
            tz = r * numpy.cos(theta)
            tR = r * numpy.sin(theta)
            out = 0.0
            for a, s, ds, H, dH in zip(
                self._Sigma_amp, self._Sigma, self._dSigmadR, self._Hz, self._dHzdz
            ):
                out += a * ds(r) * H(tz) * tR**2
                out += a * (ds(r) * H(tz) * tz / r + s(r) * dH(tz)) * tz * r
            return out * numpy.sin(theta)

        return out + 2.0 * numpy.pi * integrate.quad(_integrand, 0.0, numpy.pi)[0]


def phiME_dens(R, z, phi, dens, Sigma, dSigmadR, d2SigmadR2, hz, Hz, dHzdz, Sigma_amp):
    """The density corresponding to phi_ME"""
    r = numpy.sqrt(R**2.0 + z**2.0)
    out = dens(R, z, phi)
    for a, s, ds, d2s, h, H, dH in zip(
        Sigma_amp, Sigma, dSigmadR, d2SigmadR2, hz, Hz, dHzdz
    ):
        out -= a * (s(r) * h(z) + d2s(r) * H(z) + 2.0 / r * ds(r) * (H(z) + z * dH(z)))
    return out
