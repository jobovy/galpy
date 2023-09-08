###############################################################################
#   ChandrasekharDynamicalFrictionForce: Class that implements the
#                                        Chandrasekhar dynamical friction
###############################################################################
import copy
import hashlib

import numpy
from scipy import interpolate, special

from ..util import conversion
from .DissipativeForce import DissipativeForce
from .Potential import _check_c, evaluateDensities
from .Potential import flatten as flatten_pot

_INVSQRTTWO = 1.0 / numpy.sqrt(2.0)
_INVSQRTPI = 1.0 / numpy.sqrt(numpy.pi)


class ChandrasekharDynamicalFrictionForce(DissipativeForce):
    """Class that implements the Chandrasekhar dynamical friction force

    .. math::


       \\mathbf{F}(\\mathbf{x},\\mathbf{v}) = -2\\pi\\,[G\\,M]\\,[G\\,\\rho(\\mathbf{x})]\\,\\ln[1+\\Lambda^2] \\,\\left[\\mathrm{erf}(X)-\\frac{2X}{\\sqrt{\\pi}}\\exp\\left(-X^2\\right)\\right]\\,\\frac{\\mathbf{v}}{|\\mathbf{v}|^3}\\,

    on a mass (e.g., a satellite galaxy or a black hole) :math:`M` at position :math:`\\mathbf{x}` moving at velocity :math:`\\mathbf{v}` through a background density :math:`\\rho`. The quantity :math:`X` is the usual :math:`X=|\\mathbf{v}|/[\\sqrt{2}\\sigma_r(r)`. The factor :math:`\\Lambda` that goes into the Coulomb logarithm is taken to be

    .. math::

       \\Lambda = \\frac{r/\\gamma}{\\mathrm{max}\\left(r_{\\mathrm{hm}},GM/|\\mathbf{v}|^2\\right)}\\,,

    where :math:`\\gamma` is a constant. This :math:`\\gamma` should be the absolute value of the logarithmic slope of the density :math:`\\gamma = |\\mathrm{d} \\ln \\rho / \\mathrm{d} \\ln r|`, although for :math:`\\gamma<1` it is advisable to set :math:`\\gamma=1`. Implementation here roughly follows [2]_ and earlier work.

    """

    def __init__(
        self,
        amp=1.0,
        GMs=0.1,
        gamma=1.0,
        rhm=0.0,
        dens=None,
        sigmar=None,
        const_lnLambda=False,
        minr=0.0001,
        maxr=25.0,
        nr=501,
        ro=None,
        vo=None,
    ):
        """
        Initialize a Chandrasekhar Dynamical Friction force [1]_.

        Parameters
        ----------
        amp : float
            Amplitude to be applied to the potential (default: 1).
        GMs : float or Quantity
            Satellite mass; can be a Quantity with units of mass or Gxmass; can be adjusted after initialization by setting obj.GMs= where obj is your ChandrasekharDynamicalFrictionForce instance (note that the mass of the satellite can *not* be changed simply by multiplying the instance by a number, because he mass is not only used as an amplitude).
        rhm : float or Quantity
            Half-mass radius of the satellite (set to zero for a black hole); can be adjusted after initialization by setting obj.rhm= where obj is your ChandrasekharDynamicalFrictionForce instance.
        gamma : float
            Free-parameter in :math:`\\Lambda`.
        dens : Potential instance or list thereof, optional
            Potential instance or list thereof that represents the density [default: LogarithmicHaloPotential(normalize=1.,q=1.)].
        sigmar : callable, optional
            Function that gives the velocity dispersion as a function of r (has to be in natural units!); if None, computed from the dens potential using the spherical Jeans equation (in galpy.df.jeans) assuming zero anisotropy; if set to a lambda function, *the object cannot be pickled* (so set it to a real function).
        const_lnLambda : bool, optional
            If set to a number, use a constant ln(Lambda) instead with this value.
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
        - 2011-12-26 - Started - Bovy (NYU)
        - 2018-03-18 - Re-started: updated to r dependent Lambda form and integrated into galpy framework - Bovy (UofT)
        - 2018-07-23 - Calculate sigmar from the Jeans equation and interpolate it; allow GMs and rhm to be set on the fly - Bovy (UofT)

        References
        ----------
        .. [1] Chandrasekhar, S. (1943), Astrophysical Journal, 97, 255. ADS: http://adsabs.harvard.edu/abs/1943ApJ....97..255C.
        .. [2] Petts, J. A., Gualandris, A., Read, J. I., & Bovy, J. (2016), Monthly Notices of the Royal Astronomical Society, 463, 858. ADS: http://adsabs.harvard.edu/abs/2016MNRAS.463..858P.
        """
        DissipativeForce.__init__(self, amp=amp * GMs, ro=ro, vo=vo, amp_units="mass")
        rhm = conversion.parse_length(rhm, ro=self._ro)
        minr = conversion.parse_length(minr, ro=self._ro)
        maxr = conversion.parse_length(maxr, ro=self._ro)
        self._gamma = gamma
        self._ms = (
            self._amp / amp
        )  # from handling in __init__ above, should be ms in galpy units
        self._rhm = rhm
        self._minr = minr
        self._maxr = maxr
        self._dens_kwarg = dens  # for pickling
        self._sigmar_kwarg = sigmar  # for pickling
        # Parse density
        if dens is None:
            from .LogarithmicHaloPotential import LogarithmicHaloPotential

            dens = LogarithmicHaloPotential(normalize=1.0, q=1.0)
            if sigmar is None:  # we know this solution!
                sigmar = lambda x: _INVSQRTTWO
        dens = flatten_pot(dens)
        self._dens_pot = dens
        self._dens = lambda R, z, phi=0.0, t=0.0: evaluateDensities(
            self._dens_pot, R, z, phi=phi, t=t, use_physical=False
        )
        if sigmar is None:
            from ..df import jeans

            sigmar = lambda x: jeans.sigmar(
                self._dens_pot, x, beta=0.0, use_physical=False
            )
        self._sigmar_rs_4interp = numpy.linspace(self._minr, self._maxr, nr)
        self._sigmars_4interp = numpy.array(
            [sigmar(x) for x in self._sigmar_rs_4interp]
        )
        if numpy.any(numpy.isnan(self._sigmars_4interp)):
            # Check for case where density is zero, in that case, just
            # paint in the nearest neighbor for the interpolation
            # (doesn't matter in the end, because force = 0 when dens = 0)
            nanrs_indx = numpy.isnan(self._sigmars_4interp)
            if numpy.all(
                numpy.array(
                    [
                        self._dens(r * _INVSQRTTWO, r * _INVSQRTTWO)
                        for r in self._sigmar_rs_4interp[nanrs_indx]
                    ]
                )
                == 0.0
            ):
                self._sigmars_4interp[nanrs_indx] = interpolate.interp1d(
                    self._sigmar_rs_4interp[True ^ nanrs_indx],
                    self._sigmars_4interp[True ^ nanrs_indx],
                    kind="nearest",
                    fill_value="extrapolate",
                )(self._sigmar_rs_4interp[nanrs_indx])
        self.sigmar_orig = sigmar
        self.sigmar = interpolate.InterpolatedUnivariateSpline(
            self._sigmar_rs_4interp, self._sigmars_4interp, k=3
        )
        if const_lnLambda:
            self._lnLambda = const_lnLambda
        else:
            self._lnLambda = False
        self._amp *= 4.0 * numpy.pi
        self._force_hash = None
        self.hasC = _check_c(self._dens_pot, dens=True)
        return None

    def GMs(self, gms):
        gms = conversion.parse_mass(gms, ro=self._ro, vo=self._vo)
        self._amp *= gms / self._ms
        self._ms = gms
        # Reset the hash
        self._force_hash = None
        return None

    GMs = property(None, GMs)

    def rhm(self, new_rhm):
        self._rhm = conversion.parse_length(new_rhm, ro=self._ro)
        # Reset the hash
        self._force_hash = None
        return None

    rhm = property(None, rhm)

    def lnLambda(self, r, v):
        """
        Evaluate the Coulomb logarithm ln Lambda.

        Parameters
        ----------
        r : float
            Spherical radius (natural units).
        v : float
            Current velocity in cylindrical coordinates (natural units).

        Returns
        -------
        lnLambda : float
            Coulomb logarithm.

        Notes
        -----
        - 2018-03-18 - Started - Bovy (UofT)

        """
        if self._lnLambda:
            lnLambda = self._lnLambda
        else:
            GMvs = self._ms / v**2.0
            if GMvs < self._rhm:
                Lambda = r / self._gamma / self._rhm
            else:
                Lambda = r / self._gamma / GMvs
            lnLambda = 0.5 * numpy.log(1.0 + Lambda**2.0)
        return lnLambda

    def _calc_force(self, R, phi, z, v, t):
        r = numpy.sqrt(R**2.0 + z**2.0)
        if r < self._minr:
            self._cached_force = 0.0
        else:
            vs = numpy.sqrt(v[0] ** 2.0 + v[1] ** 2.0 + v[2] ** 2.0)
            if r > self._maxr:
                sr = self.sigmar_orig(r)
            else:
                sr = self.sigmar(r)
            X = vs * _INVSQRTTWO / sr
            Xfactor = special.erf(X) - 2.0 * X * _INVSQRTPI * numpy.exp(-(X**2.0))
            lnLambda = self.lnLambda(r, vs)
            self._cached_force = (
                -self._dens(R, z, phi=phi, t=t) / vs**3.0 * Xfactor * lnLambda
            )

    def _Rforce(self, R, z, phi=0.0, t=0.0, v=None):
        new_hash = hashlib.md5(
            numpy.array([R, phi, z, v[0], v[1], v[2], t])
        ).hexdigest()
        if new_hash != self._force_hash:
            self._calc_force(R, phi, z, v, t)
        return self._cached_force * v[0]

    def _phitorque(self, R, z, phi=0.0, t=0.0, v=None):
        new_hash = hashlib.md5(
            numpy.array([R, phi, z, v[0], v[1], v[2], t])
        ).hexdigest()
        if new_hash != self._force_hash:
            self._calc_force(R, phi, z, v, t)
        return self._cached_force * v[1] * R

    def _zforce(self, R, z, phi=0.0, t=0.0, v=None):
        new_hash = hashlib.md5(
            numpy.array([R, phi, z, v[0], v[1], v[2], t])
        ).hexdigest()
        if new_hash != self._force_hash:
            self._calc_force(R, phi, z, v, t)
        return self._cached_force * v[2]

    # Pickling functions
    def __getstate__(self):
        pdict = copy.copy(self.__dict__)
        # rm lambda function
        del pdict["_dens"]
        if self._sigmar_kwarg is None:
            # because an object set up with sigmar = user-provided function
            # cannot typically be picked, disallow this explicitly
            # (so if it can, everything should be fine; if not, pickling error)
            del pdict["sigmar_orig"]
        return pdict

    def __setstate__(self, pdict):
        self.__dict__ = pdict
        # Re-setup _dens
        self._dens = lambda R, z, phi=0.0, t=0.0: evaluateDensities(
            self._dens_pot, R, z, phi=phi, t=t, use_physical=False
        )
        # Re-setup sigmar_orig
        if self._dens_kwarg is None and self._sigmar_kwarg is None:
            self.sigmar_orig = lambda x: _INVSQRTTWO
        else:
            from ..df import jeans

            self.sigmar_orig = lambda x: jeans.sigmar(
                self._dens_pot, x, beta=0.0, use_physical=False
            )
        return None
