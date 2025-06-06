import numpy as np
import scipy.special as sp

from ..util import conversion
from .ChandrasekharDynamicalFrictionForce import ChandrasekharDynamicalFrictionForce


class FDMDynamicalFrictionForce(ChandrasekharDynamicalFrictionForce):
    def __init__(
        self,
        amp=1.0,
        GMs=0.1,
        gamma=1.0,
        rhm=0.0,
        m=1e-99,  # roughly 1e-22 eV
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
            conversion.parse_mass(m, ro=self._ro, vo=self._vo)
            / conversion._GHBARINKM3S3KPC2
            * self._ro**2
            * self._vo**3
        )
        # hasC set in ChandrasekharDynamicalFrictionForce.__init__

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
            kr = 2 * self.krValue(r, vs)
            I = -sp.sici(kr)[1] + np.log(kr) + numpy.euler_gamma

            return I + (np.sin(kr) / (kr)) - 1

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
