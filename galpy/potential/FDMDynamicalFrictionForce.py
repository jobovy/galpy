import astropy.units as u
import numpy as np
import scipy.special as sp
from astropy.constants import c, hbar
from scipy import interpolate
from scipy.integrate import quad

from . ChandrasekharDynamicalFrictionForce import ChandrasekharDynamicalFrictionForce


class FDMDynamicalFrictionForce(ChandrasekharDynamicalFrictionForce):
    def __init__(
        self,
        amp=1.0,
        GMs=0.1,
        gamma=1.0,
        rhm=0.0,
        m=1e-22 * u.eV,
        dens=None,
        sigmar=None,
        const_lnLambda=False,
        minr=0.0001,
        maxr=25.0,
        nr=501,
        ro=None,
        vo=None,
    ):
        ChandrasekharDynamicalFrictionForce.__init__(
            self,
            amp=amp,
            GMs=GMs,
            rhm=rhm,
            dens=dens,
            gamma=gamma,
            sigmar=sigmar,
            const_lnLambda=const_lnLambda,
            minr=minr,
            maxr=maxr,
            nr=nr,
            ro=ro,
            vo=vo,
        )

        self._mhbar = (
            (m / (hbar * c**2)).to(1 / (u.kpc * (u.km / u.s))).value
            * self._ro
            * self._vo
        )
        print("mhbar = ", self._mhbar)
        self._minkr = 0.0005
        self._maxkr = 300
        self._nkr = 1000
        self._kr_4interp = np.linspace(self._minkr, self._maxkr, self._nkr)
        self._integral_kr_4interp = np.array(
            [
                quad(lambda t: (1 - np.cos(t)) / t, 0, 2 * kr_i, limit=200)[0]
                for kr_i in self._kr_4interp
            ]
        )
        self._integral_kr = interpolate.InterpolatedUnivariateSpline(
            self._kr_4interp, self._integral_kr_4interp, k=3
        )

    def FDMfactor(self, r, vs):
        """
        Evaluate the FDM dynamical friction factor.
        Parameters
        ----------
        r : float
            Spherical radius (natural units).
        vs : float
            Current velocity in cylindrical coordinates (natural units).
        Returns
        -------
        FDMfactor : float
            FDM dynamical friction factor.
        """
        if self._lnLambda:
            return self._lnLambda
        else:
            kr = self.krValue(r, vs)

            if kr > self._maxkr:
                I = self._integral_kr(self._maxkr)
            else:
                I = self._integral_kr(kr)
            return I + (np.sin(2 * kr) / (2 * kr)) - 1

    def ChandraFactor(self, r, vs):
        """
        Evaluate the classical dynamical friction factor.
        Parameters
        ----------
        r : float
            Spherical radius (natural units).
        vs : float
            Current velocity in cylindrical coordinates (natural units).
        Returns
        -------
        ChandraFactor : float
            Classical dynamical friction factor.
        """

        if r > self._maxr:
            sr = self.sigmar_orig(r)
        else:
            sr = self.sigmar(r)
        X = vs / (np.sqrt(2) * sr)
        Xfactor = sp.erf(X) - 2.0 * X * (1 / np.sqrt(np.pi)) * np.exp(-(X**2.0))
        lnLambda = self.lnLambda(r, vs)

        return lnLambda * Xfactor

    def krValue(self, r, v):
        """
        Evaluate the dimensionless kr parameter kr = mrv / hbar.

        Parameters
        ----------
        r : float
            Spherical radius (natural units).
        v : float
            Current velocity in cylindrical coordinates (natural units).
        Returns
        -------
        kr : float
            Dimensionless kr parameter.
        """
        return self._mhbar * v * r

    def _calc_force(self, R, phi, z, v, t):
        r = np.sqrt(R**2.0 + z**2.0)
        if r < self._minr:
            self._cached_force = 0.0
        else:
            vs = np.sqrt(v[0] ** 2.0 + v[1] ** 2.0 + v[2] ** 2.0)
            self._C_cdm = self.ChandraFactor(r, vs)
            self._C_fdm = self.FDMfactor(r, vs)

            if self._C_fdm < self._C_cdm:
                self._cached_force = (
                    -self._dens(R, z, phi=phi, t=t) / vs**3.0 * self._C_fdm
                )
            else:
                self._cached_force = (
                    -self._dens(R, z, phi=phi, t=t) / vs**3.0 * self._C_cdm
                )
