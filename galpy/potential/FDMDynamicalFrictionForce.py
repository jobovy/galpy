import numpy
import scipy.special as sp

from ..util import conversion
from .ChandrasekharDynamicalFrictionForce import ChandrasekharDynamicalFrictionForce


class FDMDynamicalFrictionForce(ChandrasekharDynamicalFrictionForce):
    r"""
    Implements the fuzzy dark matter (FDM) dynamical friction force.

    The force is given by:

    .. math::

        \vec{F}_\mathrm{FDM} = -\frac{4\pi\mathcal{G}^2 M_\mathrm{obj}^2 \rho}{v^3} C_\mathrm{FDM}(kr) \vec{v}

    where the coefficient :math:`C_\mathrm{FDM}(kr)` depends on :math:`kr = \frac{m v r}{\hbar}` and is given by

    .. math::

        C_\mathrm{FDM}(kr) = \mathrm{Cin}(2kr) + \frac{\sin(2kr)}{2kr} - 1

    with

    .. math::

        \mathrm{Cin}(z) = \int_0^z \frac{1 - \cos(t)}{t} \, \mathrm{d}t

    For comparison, the classical (CDM) coefficient is

    .. math::

        C_\mathrm{CDM} = \ln \Lambda \left[ \mathrm{erf}(X) - \frac{2X}{\sqrt{\pi}} \exp(-X^2) \right]

    See also
    --------
    :class:`.ChandrasekharDynamicalFrictionForce`
        For the implementation and documentation of the classical Chandrasekhar dynamical friction force (CDM case).


    Since the FDM coefficient :math:`C_\mathrm{FDM}(kr)` has an asymptotic behavior, we must use the classical coefficient :math:`C_\mathrm{CDM}` as a cutoff whenever :math:`C_\mathrm{FDM}(kr) > C_\mathrm{CDM}`, because it would mean that we are in the classical regime.

    Notes
    ----------
    2025-05-30: Started (A.Szpilfidel)

    References
    ----------
    .. [1] Hui et al. (2017), https://arxiv.org/pdf/1610.08297
    """

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
        const_FDMfactor=False,
        minr=0.0001,
        maxr=25.0,
        nr=501,
        ro=None,
        vo=None,
    ):
        """
        Initialize a FDM Dynamical Friction force [1]_.

        Parameters
        ----------
        amp : float
            Amplitude to be applied to the potential (default: 1).
        GMs : float or Quantity
            Satellite mass; can be a Quantity with units of mass or Gxmass; can be adjusted after initialization by setting obj.GMs= where obj is your ChandrasekharDynamicalFrictionForce instance (note that the mass of the satellite can *not* be changed simply by multiplying the instance by a number, because he mass is not only used as an amplitude).
        rhm : float or Quantity
            Half-mass radius of the satellite (set to zero for a black hole); can be adjusted after initialization by setting obj.rhm= where obj is your ChandrasekharDynamicalFrictionForce instance.
        m : float or Quantity
            Mass of the Fuzzy Dark Matter particle; can be a Quantity with units of eV; default is set to 1e-99, which is roughly 1e-22 eV.
        gamma : float
            Free-parameter in :math:`\\Lambda`.
        dens : Potential instance or list thereof, optional
            Potential instance or list thereof that represents the density [default: LogarithmicHaloPotential(normalize=1.,q=1.)].
        sigmar : callable, optional
            Function that gives the velocity dispersion as a function of r (has to be in natural units!); if None, computed from the dens potential using the spherical Jeans equation (in galpy.df.jeans) assuming zero anisotropy; if set to a lambda function, *the object cannot be pickled* (so set it to a real function).
        const_lnLambda : bool, optional
            If set to a number, use a constant ln(Lambda) instead with this value.
        const_FDMfactor : bool, optional
            If set to a number, use a constant FDM factor instead with this value; if set to False, the FDM factor is calculated from the mass of the Fuzzy Dark Matter particle and the current radius and velocity.
        minr : float or Quantity, optional
            Minimum r at which to apply dynamical friction: at r < minr, friction is set to zero.
        maxr : float or Quantity, optional
            Maximum r for which sigmar gets interpolated; for best performance set this to the maximum r you will consider.
        nr : int, optional
            Number of radii to use in the interpolation of sigmar.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----

        References
        ----------
        .. [1] Hui and al. (2017) (https://arxiv.org/pdf/1610.08297)
        """
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
        self._const_FDMfactor = const_FDMfactor
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
        if self._const_FDMfactor:
            return self._const_FDMfactor
        else:
            kr = 2 * self.krValue(r, vs)
            I = -sp.sici(kr)[1] + numpy.log(kr) + numpy.euler_gamma

            return I + (numpy.sin(kr) / (kr)) - 1

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
        X = vs / (numpy.sqrt(2) * sr)
        Xfactor = sp.erf(X) - 2.0 * X * (1 / numpy.sqrt(numpy.pi)) * numpy.exp(
            -(X**2.0)
        )
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
        r = numpy.sqrt(R**2.0 + z**2.0)
        if r < self._minr:
            self._cached_force = 0.0
        else:
            vs = numpy.sqrt(v[0] ** 2.0 + v[1] ** 2.0 + v[2] ** 2.0)
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
