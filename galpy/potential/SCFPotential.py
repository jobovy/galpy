import hashlib
import inspect

import numpy
import scipy
from numpy.polynomial.legendre import leggauss
from packaging.version import parse as parse_version
from scipy import integrate
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy.special import gamma, gammaln

_SCIPY_VERSION = parse_version(scipy.__version__)
if _SCIPY_VERSION < parse_version("1.15"):  # pragma: no cover
    from scipy.special import lpmn
else:
    from scipy.special import assoc_legendre_p_all

from ..backend import (
    as_numpy,
    asarray_on_device,
    coerce_coords,
    device_of,
    get_namespace,
    is_backend_array,
    like,
    match_input_dtype,
)
from ..backend import use as _use_backend
from ..backend.special import assoc_legendre, gegenbauer
from ..util import conversion, coords
from ..util._optional_deps import _APY_LOADED
from ..util._pickle import SplinePickleMixin
from ..util.special import compute_legendre, sph_harm_normalization
from .Potential import Potential

if _APY_LOADED:
    from astropy import units

from .SphericalHarmonicPotentialMixin import SphericalHarmonicPotentialMixin


class SCFPotential(Potential, SphericalHarmonicPotentialMixin, SplinePickleMixin):
    """Class that implements the `Hernquist & Ostriker (1992) <http://adsabs.harvard.edu/abs/1992ApJ...386..375H>`_ Self-Consistent-Field-type potential.
    Note that we divide the amplitude by 2 such that :math:`Acos = \\delta_{0n}\\delta_{0l}\\delta_{0m}` and :math:`Asin = 0` corresponds to :ref:`Galpy's Hernquist Potential <hernquist_potential>`.

    .. math::

        \\rho(r, \\theta, \\phi) = \\frac{amp}{2}\\sum_{n=0}^{\\infty} \\sum_{l=0}^{\\infty} \\sum_{m=0}^l N_{lm} P_{lm}(\\cos(\\theta))  \\tilde{\\rho}_{nl}(r) \\left(A_{cos, nlm} \\cos(m\\phi) + A_{sin, nlm} \\sin(m\\phi)\\right)

    where

    .. math::

        \\tilde{\\rho}_{nl}(r) = \\frac{K_{nl}}{\\sqrt{\\pi}} \\frac{(a r)^l}{(r/a) (a + r)^{2l + 3}} C_{n}^{2l + 3/2}(\\xi)
    .. math::

        \\Phi(r, \\theta, \\phi) = \\sum_{n=0}^{\\infty} \\sum_{l=0}^{\\infty} \\sum_{m=0}^l N_{lm} P_{lm}(\\cos(\\theta))  \\tilde{\\Phi}_{nl}(r) \\left(A_{cos, nlm} \\cos(m\\phi) + A_{sin, nlm} \\sin(m\\phi)\\right)

    where

    .. math::
        \\tilde{\\Phi}_{nl}(r) = -\\sqrt{4 \\pi}K_{nl} \\frac{(ar)^l}{(a + r)^{2l + 1}} C_{n}^{2l + 3/2}(\\xi)


    where

    .. math::

        \\xi = \\frac{r - a}{r + a} \\qquad
        N_{lm} = \\sqrt{\\frac{2l + 1}{4\\pi} \\frac{(l - m)!}{(l + m)!}}(2 - \\delta_{m0}) \\qquad
        K_{nl} = \\frac{1}{2} n (n + 4l + 3) + (l + 1)(2l + 1)

    and :math:`P_{lm}` is the Associated Legendre Polynomials whereas :math:`C_n^{\\alpha}` is the Gegenbauer polynomial.

    **Time-dependent potentials** are supported by letting each expansion
    coefficient :math:`A_{\\cos,nlm}` and :math:`A_{\\sin,nlm}` be a function of
    time. This is enabled by passing a ``tgrid`` array together with either

    * ``Acos`` (and optionally ``Asin``) as callables ``f(t)`` returning the
      ``(N,L,M)`` coefficient array at time ``t``, or
    * ``Acos``/``Asin`` as precomputed ``(Nt,N,L,M)`` arrays sampled on ``tgrid``,

    or via ``from_density`` by passing a density that depends on time (i.e.,
    ``dens(R, z, phi, t=0.)``) together with a ``tgrid``. In all cases the
    coefficients are sampled on ``tgrid`` and interpolated in time with a cubic
    spline, allowing efficient evaluation of the potential, forces, second
    derivatives, and density at arbitrary times within (or, by extrapolation,
    outside) the ``tgrid`` range in both Python and C (for orbit integration).
    """

    # Cubic-spline time interpolators over the coefficient arrays; created only
    # by the time-dependent branch, so SplinePickleMixin skips them when absent.
    _PICKLE_SPLINE_ATTRS = ("_Acos_interp", "_Asin_interp")

    def __init__(
        self,
        amp=1.0,
        Acos=numpy.array([[[1]]]),
        Asin=None,
        a=1.0,
        tgrid=None,
        normalize=False,
        ro=None,
        vo=None,
    ):
        """
        Initialize a SCF Potential from a set of expansion coefficients (use SCFPotential.from_density to directly initialize from a density)

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass.
        Acos : numpy.ndarray or callable, optional
            The real part of the expansion coefficient (NxLxL matrix, or optionally NxLx1 if Asin=None). For a time-dependent potential (``tgrid`` given), this is instead either a callable ``f(t)`` returning such an (N,L,L) / (N,L,1) array, or a precomputed (Nt,N,L,L) / (Nt,N,L,1) array sampled on ``tgrid``.
        Asin : numpy.ndarray or callable, optional
            The imaginary part of the expansion coefficient (NxLxL matrix or None). For a time-dependent potential, either a callable ``f(t)`` or a precomputed (Nt,N,L,L) array (or None for an axisymmetric potential).
        a : float or Quantity, optional
            Scale length.
        tgrid : numpy.ndarray or None, optional
            Time grid for time-dependent potentials. If provided, ``Acos`` and ``Asin`` are interpreted as time-dependent coefficients (callables ``f(t)`` or arrays sampled on ``tgrid``): each coefficient is sampled on ``tgrid`` and interpolated in time with a cubic spline, allowing fast evaluation (in both Python and C) at arbitrary times within the ``tgrid`` range. Default: ``None`` (static potential).
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2016-05-13 - Written - Aladdin Seaifan (UofT)
        - 2026-07-02 - Added time-dependent support via tgrid - Bovy (UofT)
        """
        Potential.__init__(self, amp=amp / 2.0, ro=ro, vo=vo, amp_units="mass")
        a = conversion.parse_length(a, ro=self._ro)
        self._a = a
        self._tdep = tgrid is not None
        if self._tdep:
            self._init_timedep(Acos, Asin, tgrid)
        else:
            self._init_static(Acos, Asin)
        self._force_hash = None
        self._2nd_deriv_cache_key = None
        self._cached_2nd_derivs = None
        self.hasC = True
        self._backend_compatible = True
        self.hasC_dxdv = True
        # Full 3D Hessian (R2/z2/Rz/phi2/Rphi/zphi deriv) is implemented in C
        # via the spherical-harmonic expansion, so 3D variational (dxdv)
        # integration is supported.
        self.hasC_dxdv3d = True
        self.hasC_dens = True
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):
            self.normalize(normalize)
        return None

    def _init_static(self, Acos, Asin):
        """
        Set up a static (time-independent) SCFPotential from coefficient arrays.

        Notes
        -----
        - 2016-05-13 - Written - Aladdin Seaifan (UofT)
        - 2026-07-02 - Split out of __init__ for time-dependent support - Bovy (UofT)
        """
        ##Errors
        shape = Acos.shape
        # Validation and symmetry detection are DISCRETE structural decisions
        # (shape errors, the m>l warning, isNonAxi) -- not part of the
        # differentiable computation -- and numpy.triu/all/any reject a backend
        # array. Read them off numpy views; the coefficients STORED below stay
        # in whatever namespace they arrived in, so a backend build keeps its
        # gradient. No-op on numpy.
        _Ac = as_numpy(Acos)
        _As = None if Asin is None else as_numpy(Asin)
        errorMessage = None
        if len(shape) != 3:
            errorMessage = "Acos must be a 3 dimensional numpy array"
        elif Asin is not None and shape[1] != shape[2]:
            errorMessage = "The second and third dimension of the expansion coefficients must have the same length"
        elif Asin is None and not (shape[2] == 1 or shape[1] == shape[2]):
            errorMessage = "The third dimension must have length=1 or equal to the length of the second dimension"
        elif Asin is None and shape[1] > 1 and numpy.any(_Ac[:, :, 1:] != 0):
            errorMessage = (
                "Acos has non-zero elements at indices m>0, which implies a non-axi symmetric potential.\n"
                + "Asin=None which implies an axi symmetric potential.\n"
                + "Contradiction."
            )
        elif Asin is not None and Asin.shape != shape:
            errorMessage = "The shape of Asin does not match the shape of Acos."
        if errorMessage is not None:
            raise RuntimeError(errorMessage)

        ##Warnings
        warningMessage = None
        if numpy.any(numpy.triu(_Ac, 1) != 0) or (
            _As is not None and numpy.any(numpy.triu(_As, 1) != 0)
        ):
            warningMessage = (
                "Found non-zero values at expansion coefficients where m > l\n"
                + "The Mth and Lth dimension is expected to make a lower triangular matrix.\n"
                + "All values found above the diagonal will be ignored."
            )
        if warningMessage is not None:
            raise RuntimeWarning(warningMessage)

        ##Is non axi?
        self.isNonAxi = True
        if (
            Asin is None
            or shape[1] == 1
            or (numpy.all(_Ac[:, :, 1:] == 0) and numpy.all(_As[:, :, :] == 0))
        ):
            self.isNonAxi = False

        NN = sph_harm_normalization(Acos.shape[1], Acos.shape[2])

        self._Acos = Acos * NN[numpy.newaxis, :, :]
        if Asin is not None:
            self._Asin = Asin * NN[numpy.newaxis, :, :]
        else:
            self._Asin = numpy.zeros_like(Acos)

    @staticmethod
    def _coeffs_to_timeseries(coeffs, tgrid, name):
        """
        Turn time-dependent coefficient input into an (Nt,N,L,M) array.

        ``coeffs`` is either a callable ``f(t)`` returning an (N,L,M) array (sampled
        on ``tgrid``) or a precomputed (Nt,N,L,M) array. Full shape validation is
        performed by the caller (``_init_timedep``).

        Notes
        -----
        - 2026-07-02 - Written - Bovy (UofT)
        """
        if callable(coeffs):
            arr = numpy.array([numpy.asarray(coeffs(t), dtype=float) for t in tgrid])
        else:
            arr = numpy.asarray(coeffs, dtype=float)
        return arr

    def _init_timedep(self, Acos, Asin, tgrid):
        """
        Set up a time-dependent SCFPotential.

        Each coefficient ``A_nlm`` (both cos and sin) is sampled on ``tgrid`` and
        interpolated in time with a cubic spline. The interpolants are used both
        for Python evaluation (via ``_ensure_coeffs_for_time``) and for the C
        implementation (via ``_parse_scf_pot``), guaranteeing Python/C parity.

        Notes
        -----
        - 2026-07-02 - Written - Bovy (UofT)
        """
        self._tgrid = numpy.asarray(tgrid, dtype=float)
        Nt = len(self._tgrid)
        Acos_all = self._coeffs_to_timeseries(Acos, self._tgrid, "Acos")
        Asin_all = (
            self._coeffs_to_timeseries(Asin, self._tgrid, "Asin")
            if Asin is not None
            else None
        )
        ##Errors (each time slice must satisfy the static coefficient constraints)
        shape = Acos_all.shape
        # Same discrete validation/symmetry gate as _init_static.
        _Ac = as_numpy(Acos_all)
        _As = None if Asin_all is None else as_numpy(Asin_all)
        errorMessage = None
        if Acos_all.ndim != 4 or shape[0] != Nt:
            errorMessage = (
                "For a time-dependent SCFPotential, Acos must be a callable f(t) "
                "returning a 3 dimensional (N,L,M) array, or a 4 dimensional "
                "(Nt,N,L,M) array sampled on tgrid (with Nt matching len(tgrid))"
            )
        elif Asin_all is not None and shape[2] != shape[3]:
            errorMessage = "The second and third dimension of the expansion coefficients must have the same length"
        elif Asin_all is None and not (shape[3] == 1 or shape[2] == shape[3]):
            errorMessage = "The third dimension must have length=1 or equal to the length of the second dimension"
        elif Asin_all is None and shape[2] > 1 and numpy.any(_Ac[:, :, :, 1:] != 0):
            errorMessage = (
                "Acos has non-zero elements at indices m>0, which implies a non-axi symmetric potential.\n"
                + "Asin=None which implies an axi symmetric potential.\n"
                + "Contradiction."
            )
        elif Asin_all is not None and Asin_all.shape != shape:
            errorMessage = "The shape of Asin does not match the shape of Acos."
        if errorMessage is not None:
            raise RuntimeError(errorMessage)

        ##Warnings
        warningMessage = None
        if numpy.any(numpy.triu(_Ac, 1) != 0) or (
            _As is not None and numpy.any(numpy.triu(_As, 1) != 0)
        ):
            warningMessage = (
                "Found non-zero values at expansion coefficients where m > l\n"
                + "The Mth and Lth dimension is expected to make a lower triangular matrix.\n"
                + "All values found above the diagonal will be ignored."
            )
        if warningMessage is not None:
            raise RuntimeWarning(warningMessage)

        ##Is non axi? (checked over all times)
        self.isNonAxi = True
        if (
            Asin_all is None
            or shape[2] == 1
            or (numpy.all(_Ac[:, :, :, 1:] == 0) and numpy.all(_As == 0))
        ):
            self.isNonAxi = False

        N, L, M = shape[1], shape[2], shape[3]
        NN = sph_harm_normalization(L, M)
        self._Acos_all = Acos_all * NN[numpy.newaxis, numpy.newaxis, :, :]
        if Asin_all is not None:
            self._Asin_all = Asin_all * NN[numpy.newaxis, numpy.newaxis, :, :]
        else:
            self._Asin_all = numpy.zeros_like(self._Acos_all)
        self._coeff_shape = (N, L, M)
        # Cubic-spline time interpolators over the flattened coefficient arrays;
        # CubicSpline is a PPoly subclass whose coefficients are passed to C for
        # exact Python/C parity (see _parse_scf_pot).
        self._Acos_interp = CubicSpline(self._tgrid, self._Acos_all.reshape(Nt, -1))
        self._Asin_interp = CubicSpline(self._tgrid, self._Asin_all.reshape(Nt, -1))
        self._cached_coeff_t = None
        # Placeholder current-time coefficients; refreshed by
        # _ensure_coeffs_for_time before each evaluation.
        self._Acos = self._Acos_all[0]
        self._Asin = self._Asin_all[0]

    def _ensure_coeffs_for_time(self, t):
        """
        Refresh ``self._Acos``/``self._Asin`` to the cubic-spline-interpolated
        coefficients at time ``t`` (no-op for static potentials).

        On the numpy path this evaluates the two scipy ``CubicSpline``
        interpolators at the float time ``t`` and caches the result (byte-identical
        to the original implementation). A backend (jax/torch) ``t`` is a no-op
        here: the backend evaluation paths obtain their (differentiable-in-``t``)
        coefficients from :meth:`_coeffs_at_time` instead, which evaluates the same
        piecewise cubic through the active namespace (so no ``float(t)`` cast and no
        mutable float-cache, which a jax tracer / a time array would break).

        Notes
        -----
        - 2026-07-02 - Written - Bovy (UofT)
        - 2026-07-07 - Namespace-dispatched (backend t handled by _coeffs_at_time)
          - Bovy (UofT)
        """
        if not self._tdep or is_backend_array(t):
            return
        tf = float(t)
        if self._cached_coeff_t == tf:
            return
        N, L, M = self._coeff_shape
        self._Acos = self._Acos_interp(tf).reshape(N, L, M)
        self._Asin = self._Asin_interp(tf).reshape(N, L, M)
        self._cached_coeff_t = tf

    def _coeffs_at_time(self, t, xp, dev):
        """
        Backend coefficient provider: return ``(Acos, Asin)`` for the active
        namespace ``xp`` on device ``dev``.

        For a static potential these are the fixed expansion tables. For a
        time-dependent potential the two scipy ``CubicSpline`` interpolators (a
        ``PPoly`` with knots ``self._tgrid`` and piecewise-cubic power-basis
        coefficients) are evaluated at ``t`` *through ``xp``* -- a
        ``searchsorted`` interval lookup plus Horner over the cubic -- so the
        result is exactly differentiable in ``t`` (matching the numpy scipy
        evaluation to ~1 ulp). A scalar ``t`` yields ``(N, L, M)`` tables; a
        ``(P,)`` time array yields ``(P, N, L, M)`` per-point tables (used by the
        batched, array-``t`` evaluation path).

        Notes
        -----
        - 2026-07-07 - Written - Bovy (UofT)
        """
        if not self._tdep:
            return (
                asarray_on_device(xp, self._Acos, dev),
                asarray_on_device(xp, self._Asin, dev),
            )
        N, L, M = self._coeff_shape
        tb = asarray_on_device(xp, t, dev)
        acos_flat = _interp_ppoly_vec(
            xp, self._Acos_interp.x, self._Acos_interp.c, tb, dev
        )
        asin_flat = _interp_ppoly_vec(
            xp, self._Asin_interp.x, self._Asin_interp.c, tb, dev
        )
        shape = (N, L, M) if getattr(tb, "ndim", 0) == 0 else (tb.shape[0], N, L, M)
        return xp.reshape(acos_flat, shape), xp.reshape(asin_flat, shape)

    @classmethod
    def from_density(
        cls,
        dens,
        N,
        L=None,
        a=1.0,
        symmetry=None,
        tgrid=None,
        radial_order=None,
        costheta_order=None,
        phi_order=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize an SCF Potential from a given density.

        Parameters
        ----------
        dens : function
            Density function that takes parameters R, z and phi; z and phi are optional for spherical profiles, phi is optional for axisymmetric profiles. The density function must take input positions in internal units (R/ro, z/ro), but can return densities in physical units. You can use the member dens of Potential instances or the density from evaluateDensities. For a time-dependent potential (``tgrid`` given), the density may additionally accept a ``t`` keyword argument (e.g., ``dens(R, z, phi, t=0.)``) or be a galpy ``Potential`` instance whose density is time-dependent.
        N : int
            Number of radial basis functions.
        L : int, optional
            Number of costheta basis functions; for non-axisymmetric profiles also sets the number of azimuthal (phi) basis functions to M = 2L+1).
        a : float or Quantity, optional
            Expansion scale length.
        symmetry : {'spherical','axisymmetry',None}, optional
            Symmetry of the profile to assume. None is the general, non-axisymmetric case.
        tgrid : numpy.ndarray, Quantity, or None, optional
            Time grid for time-dependent potentials (a Quantity in physical time units, e.g. Gyr, is accepted). If provided, the expansion coefficients are computed at each time in ``tgrid`` (passing ``t`` to the density function when it accepts one) and interpolated in time, producing a time-dependent SCFPotential. Default: ``None`` (static potential; any ``t`` argument of the density is ignored).
        radial_order : int, optional
            Number of sample points for the radial integral. If None, radial_order=max(20, N + 3/2L + 1).
        costheta_order : int, optional
            Number of sample points of the costheta integral. If None, If costheta_order=max(20, L + 1).
        phi_order : int, optional
            Number of sample points of the phi integral. If None, If costheta_order=max(20, L + 1).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Returns
        -------
        SCFPotential object

        Notes
        -----
        - Written - Jo Bovy (UofT) - 2022-06-20
        - 2026-07-02 - Added time-dependent support via tgrid - Bovy (UofT)

        """
        # Dummy object for ro/vo handling, to ensure consistency
        dumm = cls(ro=ro, vo=vo)
        internal_ro = dumm._ro
        internal_vo = dumm._vo
        a = conversion.parse_length(a, ro=internal_ro)
        if tgrid is not None:
            tgrid = conversion.parse_time(tgrid, ro=internal_ro, vo=internal_vo)
            return cls._from_density_timedep(
                dens,
                N,
                L,
                a,
                symmetry,
                tgrid,
                radial_order,
                costheta_order,
                phi_order,
                ro,
                vo,
            )
        Acos, Asin = cls._symmetry_coeffs(
            dens, N, L, a, symmetry, radial_order, costheta_order, phi_order
        )
        # Turn on physical outputs if input density was physical
        if _APY_LOADED:
            # First need to determine number of parameters, like in
            # scf_compute_coeffs_spherical/axi. Pin to numpy: the arity probe
            # calls the (backend-aware) density on plain floats, which must
            # dispatch on numpy regardless of any forced backend default.
            with _use_backend("numpy", force=True):
                numOfParam = 0
                try:
                    dens(0)
                    numOfParam = 1
                except:
                    try:
                        dens(0, 0)
                        numOfParam = 2
                    except:
                        numOfParam = 3
                param = [1] * numOfParam
                try:
                    dens(*param).to(units.kg / units.m**3)
                except (AttributeError, units.UnitConversionError, TypeError):
                    # We'll just assume that unit conversion means density
                    # is scalar Quantity. TypeError: a backend (torch) tensor
                    # has a .to() that rejects a unit -> treat as non-physical.
                    pass
                else:
                    ro = internal_ro
                    vo = internal_vo
        return cls(Acos=Acos, Asin=Asin, a=a, ro=ro, vo=vo)

    @classmethod
    def from_nbody(
        cls,
        pos,
        N,
        L=None,
        mass=1.0,
        a=1.0,
        symmetry=None,
        tgrid=None,
        ro=None,
        vo=None,
    ):
        """
        Initialize an SCFPotential from an N-body / particle representation.

        Computes the expansion coefficients directly from a set of particle
        positions and masses (using ``scf_compute_coeffs_spherical_nbody`` and
        its axisymmetric and general counterparts). A time-dependent potential is
        built by passing multiple snapshots: give ``pos`` with shape ``[3,n,nt]``
        together with a ``tgrid`` of length ``nt``, and the coefficients are
        computed at each snapshot and interpolated in time (analogous to the
        time-dependent ``from_density``). The particle sum is accumulated in
        batches so that building from a very large number of particles stays
        memory-bounded.

        Parameters
        ----------
        pos : numpy.ndarray or Quantity
            Positions of the particles in rectangular coordinates, with shape
            ``[3,n]`` (static) or ``[3,n,nt]`` (time-dependent, one snapshot per
            time in ``tgrid``).
        N : int
            Number of radial basis functions.
        L : int, optional
            Number of costheta basis functions; for non-axisymmetric profiles also
            sets the number of azimuthal (phi) basis functions to M = 2L+1.
            Required unless ``symmetry='spherical'``.
        mass : float, numpy.ndarray, or Quantity, optional
            Particle masses: a scalar (all equal), an array of shape ``[n]``, or,
            for the time-dependent case, an array of shape ``[n,nt]``. Default 1.0.
        a : float or Quantity, optional
            Expansion scale length.
        symmetry : {'spherical','axisymmetry',None}, optional
            Symmetry to assume. None is the general, non-axisymmetric case.
        tgrid : numpy.ndarray, Quantity, or None, optional
            Time grid for a time-dependent potential (a Quantity in physical time
            units, e.g. Gyr, is accepted). If provided, ``pos`` must have shape
            ``[3,n,len(tgrid)]`` and the coefficients are computed at each snapshot
            and interpolated in time. Default: ``None`` (static).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Returns
        -------
        SCFPotential object

        Notes
        -----
        - 2026-07-04 - Written - Bovy (UofT)

        """
        # Dummy object for ro/vo handling, to ensure consistency
        dumm = cls(ro=ro, vo=vo)
        internal_ro = dumm._ro
        internal_vo = dumm._vo
        # Physical outputs if any physical (Quantity) input was given
        physical = _APY_LOADED and (
            isinstance(mass, units.Quantity) or isinstance(pos, units.Quantity)
        )
        a = conversion.parse_length(a, ro=internal_ro)
        pos = numpy.asarray(conversion.parse_length(pos, ro=internal_ro), dtype=float)
        mass = numpy.asarray(
            conversion.parse_mass(mass, ro=internal_ro, vo=internal_vo), dtype=float
        )
        if pos.ndim not in (2, 3) or pos.shape[0] != 3:
            raise ValueError(
                "pos must have shape [3,n] (static) or [3,n,nt] (time-dependent)"
            )
        out_ro, out_vo = (internal_ro, internal_vo) if physical else (ro, vo)
        if tgrid is None:
            if pos.ndim != 2:
                raise ValueError("pos must have shape [3,n] when tgrid is not given")
            mass = _nbody_parse_mass(mass, pos.shape[1])
            Acos, Asin = _batched_nbody(pos, N, L, mass, a, symmetry)
            return cls(Acos=Acos, Asin=Asin, a=a, ro=out_ro, vo=out_vo)
        tgrid = numpy.asarray(
            conversion.parse_time(tgrid, ro=internal_ro, vo=internal_vo)
        )
        Nt = len(tgrid)
        if pos.ndim != 3 or pos.shape[2] != Nt:
            raise ValueError(
                "pos must have shape [3,n,nt] with nt=len(tgrid) when tgrid is given"
            )
        n = pos.shape[1]
        mass2d = mass.ndim == 2
        if mass2d and mass.shape != (n, Nt):
            raise ValueError("a 2D mass must have shape [n,nt]")
        Acos_list = []
        Asin_list = []
        any_sin = False
        for it in range(Nt):
            mass_it = _nbody_parse_mass(mass[:, it] if mass2d else mass, n)
            Ac, As = _batched_nbody(pos[:, :, it], N, L, mass_it, a, symmetry)
            Acos_list.append(Ac)
            if As is not None:
                any_sin = True
            Asin_list.append(As)
        Acos_all = numpy.array(Acos_list)
        Asin_all = numpy.array(Asin_list) if any_sin else None
        return cls(Acos=Acos_all, Asin=Asin_all, a=a, tgrid=tgrid, ro=out_ro, vo=out_vo)

    @classmethod
    def from_multipole(cls, mult, N, a=1.0, radial_order=None, ro=None, vo=None):
        """
        Initialize an SCFPotential from a MultipoleExpansionPotential.

        Because both potentials expand the density in the same real spherical
        harmonics, the translation is purely radial: the density multipoles
        rho_lm(r) of the multipole expansion are projected onto the SCF radial
        basis (a set of 1D radial integrals), with no angular quadrature. The
        angular resolution (``L``, ``M``) is taken from the multipole expansion;
        ``N`` sets the number of SCF radial basis functions. A time-dependent
        multipole expansion (built on a ``tgrid``) produces a time-dependent
        SCFPotential on the same ``tgrid``.

        Parameters
        ----------
        mult : MultipoleExpansionPotential
            The multipole expansion to translate.
        N : int
            Number of radial basis functions of the SCF expansion.
        a : float or Quantity, optional
            SCF expansion scale length.
        radial_order : int, optional
            Number of sample points for the radial projection integral. If None,
            ``max(2*N+L, 200)``.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Returns
        -------
        SCFPotential object

        Notes
        -----
        - 2026-07-04 - Written - Bovy (UofT)

        """
        dumm = cls(ro=ro, vo=vo)
        a = conversion.parse_length(a, ro=dumm._ro)
        L, M = mult._L, mult._M
        nonaxi = mult.isNonAxi
        beta = sph_harm_normalization(L, M)
        beta_safe = numpy.where(beta > 0, beta, 1.0)

        def _snapshot(t):
            # beta-baked SCF coefficients (N,L,M) from the multipole's density
            # multipoles at time t (t=None for a static multipole)
            Acos_b, Asin_b = _scf_coeffs_from_multipole(
                mult, N, L, M, a, radial_order, t=t
            )
            raw_cos = numpy.where(beta[None] > 0, Acos_b / beta_safe[None], 0.0)
            raw_sin = numpy.where(beta[None] > 0, Asin_b / beta_safe[None], 0.0)
            if nonaxi:  # SCF stores (N,L,L); pad m>=M with zeros
                pcos = numpy.zeros((N, L, L))
                psin = numpy.zeros((N, L, L))
                pcos[:, :, :M] = raw_cos
                psin[:, :, :M] = raw_sin
                return pcos, psin
            return raw_cos[:, :, :1], None  # axisymmetric: only m=0

        # SCF __init__ halves amp, so pass 2*amp to preserve the multipole's amp
        amp = 2.0 * mult._amp
        if not mult._tdep:
            Acos, Asin = _snapshot(None)
            return cls(amp=amp, Acos=Acos, Asin=Asin, a=a, ro=ro, vo=vo)
        tgrid = mult._tgrid
        Acos_list = []
        Asin_list = []
        any_sin = False
        for t in tgrid:
            Ac, As = _snapshot(t)
            Acos_list.append(Ac)
            if As is not None:
                any_sin = True
            Asin_list.append(As)
        Acos_all = numpy.array(Acos_list)
        Asin_all = numpy.array(Asin_list) if any_sin else None
        return cls(
            amp=amp, Acos=Acos_all, Asin=Asin_all, a=a, tgrid=tgrid, ro=ro, vo=vo
        )

    @staticmethod
    def _symmetry_coeffs(
        dens, N, L, a, symmetry, radial_order, costheta_order, phi_order
    ):
        """
        Compute (Acos, Asin) for a density given the assumed symmetry.

        Notes
        -----
        - 2026-07-02 - Written (factored from from_density) - Bovy (UofT)
        """
        if not symmetry is None and symmetry.startswith("spher"):
            return scf_compute_coeffs_spherical(dens, N, a=a, radial_order=radial_order)
        elif not symmetry is None and symmetry.startswith("axi"):
            return scf_compute_coeffs_axi(
                dens,
                N,
                L,
                a=a,
                radial_order=radial_order,
                costheta_order=costheta_order,
            )
        else:
            return scf_compute_coeffs(
                dens,
                N,
                L,
                a=a,
                radial_order=radial_order,
                costheta_order=costheta_order,
                phi_order=phi_order,
            )

    @classmethod
    def _from_density_timedep(
        cls,
        dens,
        N,
        L,
        a,
        symmetry,
        tgrid,
        radial_order,
        costheta_order,
        phi_order,
        ro,
        vo,
    ):
        """
        Build a time-dependent SCFPotential from a (possibly time-dependent)
        density by computing the expansion coefficients at each time in ``tgrid``.

        When the density accepts a ``t`` argument, the coefficients at all times
        are computed with a single, time-vectorized quadrature (the
        time-independent basis functions are evaluated only once rather than once
        per time step, mirroring the time-dependent ``MultipoleExpansionPotential``
        and giving a large speed-up); a density that cannot be evaluated as an
        array over ``t`` falls back to a per-timestep loop. A density without a
        ``t`` argument is treated as constant in time (the coefficients are
        computed once and broadcast). As for the time-dependent
        ``MultipoleExpansionPotential``, astropy ``Quantity`` (physical-unit)
        densities are not supported here; pass the density in galpy's internal
        units.

        Notes
        -----
        - 2026-07-02 - Written - Bovy (UofT)
        - 2026-07-02 - Vectorized over time - Bovy (UofT)
        """
        tgrid = numpy.asarray(tgrid)
        Nt = len(tgrid)
        # A galpy Potential instance -> use its (possibly time-dependent) density
        if isinstance(dens, Potential):
            dens = dens.dens
        has_t = "t" in inspect.signature(dens).parameters
        if not has_t:
            # Constant in time: compute the coefficients once and broadcast
            Acos, Asin = cls._symmetry_coeffs(
                dens, N, L, a, symmetry, radial_order, costheta_order, phi_order
            )
            Acos_all = numpy.repeat(Acos[numpy.newaxis], Nt, axis=0)
            Asin_all = (
                numpy.repeat(Asin[numpy.newaxis], Nt, axis=0)
                if Asin is not None
                else None
            )
            return cls(Acos=Acos_all, Asin=Asin_all, a=a, tgrid=tgrid, ro=ro, vo=vo)
        try:
            # Fast path: evaluate the density at all times at once, batching over
            # the tgrid to keep the (time-vectorized) working set memory-bounded
            if symmetry is not None and symmetry.startswith("spher"):
                Acos_all, Asin_all = _batched_timedep(
                    tgrid,
                    N,
                    lambda tg: _scf_compute_coeffs_spherical_timedep(
                        dens, N, tg, a=a, radial_order=radial_order
                    ),
                )
            elif symmetry is not None and symmetry.startswith("axi"):
                Acos_all, Asin_all = _batched_timedep(
                    tgrid,
                    N * L,
                    lambda tg: _scf_compute_coeffs_axi_timedep(
                        dens,
                        N,
                        L,
                        tg,
                        a=a,
                        radial_order=radial_order,
                        costheta_order=costheta_order,
                    ),
                )
            else:
                Acos_all, Asin_all = _batched_timedep(
                    tgrid,
                    2 * N * L * L,
                    lambda tg: _scf_compute_coeffs_timedep(
                        dens,
                        N,
                        L,
                        tg,
                        a=a,
                        radial_order=radial_order,
                        costheta_order=costheta_order,
                        phi_order=phi_order,
                    ),
                )
        except _TimeDepDensityNotVectorized:
            # Fall back to a per-timestep loop for densities that cannot be
            # evaluated as an array over t
            Acos_list = []
            Asin_list = []
            any_sin = False
            for t in tgrid:
                make_dens_t = lambda *args, _t=t, **kwargs: dens(*args, t=_t, **kwargs)
                Ac, As = cls._symmetry_coeffs(
                    make_dens_t,
                    N,
                    L,
                    a,
                    symmetry,
                    radial_order,
                    costheta_order,
                    phi_order,
                )
                Acos_list.append(Ac)
                if As is not None:
                    any_sin = True
                Asin_list.append(As)
            Acos_all = numpy.array(Acos_list)
            Asin_all = numpy.array(Asin_list) if any_sin else None
        return cls(Acos=Acos_all, Asin=Asin_all, a=a, tgrid=tgrid, ro=ro, vo=vo)

    def _rhoTilde(self, r, N, L):
        """
        Evaluate rho_tilde as defined in equation 3.9 and 2.24 for 0 <= n < N and 0 <= l < L

        Parameters
        ----------
        r : float
            Evaluate at radius r
        N : int
            size of the N dimension
        L : int
            size of the L dimension

        Returns
        -------
        numpy.ndarray
            The value of rho tilde

        Notes
        -----
         - Written on 2016-05-17 by Aladdin Seaifan (UofT)
        """
        xp = get_namespace(r)
        if xp is numpy:
            return _rhoTilde_basis(r, N, L, self._a)
        # backend path: coerce r first (a direct call may pass a python/numpy
        # scalar) so _RToxi/xp ops below see a backend array. Same expression,
        # functional (no in-place writes); the constant (n,l) grids are built in
        # numpy and converted once, on the input's device (CUDA support). For a
        # scalar r this returns (N, L); for an array r of shape (P,) (the
        # vectorized eval path) K gains a trailing point axis (N, L, 1) and l
        # moves to (L, 1) so the r-prefactor is (L, P)
        # and the result is (N, L, P) -- one batched call, no per-point loop.
        (r,) = coerce_coords(xp, r)
        xi = _RToxi(r, self._a)
        CC = _C(xi, N, L)
        a = self._a
        n = numpy.arange(0, N, dtype=float)[:, numpy.newaxis]
        l = numpy.arange(0, L, dtype=float)[numpy.newaxis, :]
        K = 0.5 * n * (n + 4 * l + 3) + (l + 1.0) * (2 * l + 1)
        dev = device_of(r)
        if getattr(r, "ndim", 0) != 0:
            K = asarray_on_device(xp, K[:, :, numpy.newaxis], dev)
            l = asarray_on_device(xp, l.reshape(L, 1), dev)
        else:
            K = asarray_on_device(xp, K, dev)
            l = asarray_on_device(xp, l, dev)
        return (
            K
            * ((a * r) ** l)
            / ((r / a) * (a + r) ** (2 * l + 3.0))
            * CC
            * (numpy.pi) ** -0.5
        )

    def _phiTilde(self, r, N, L):
        """
        Evaluate phi_tilde as defined in equation 3.10 and 2.25 for 0 <= n < N and 0 <= l < L

        Parameters
        ----------
        r : float
            Evaluate at radius r
        N : int
            size of the N dimension
        L : int
            size of the L dimension

        Returns
        -------
        numpy.ndarray
            phi tilde

        Notes
        -----
        - Written on 2016-05-17 by Aladdin Seaifan (UofT)

        """
        xp = get_namespace(r)
        if xp is numpy:
            return _phiTilde_basis(r, N, L, self._a)
        # backend path: branchless r == 0 handling. Both xp.where branches are
        # evaluated under tracing/eager AD, so the generic branch is computed at
        # a guarded rsafe (the r == 0 column then takes the -CC/a limit instead).
        # For a scalar r the l index sits on axis 1 (-> result (N, L)); for an
        # array r of shape (P,) (the vectorized eval path) it sits on axis 0 so
        # the r-dependent prefactor broadcasts to (L, P) and, against CC (N, L, P),
        # gives (N, L, P) -- one batched call instead of a per-point Python loop.
        # Coerce r: a direct call may pass a python/numpy scalar that xp.where rejects.
        (r,) = coerce_coords(xp, r)
        xi = _RToxi(r, self._a)
        CC = _C(xi, N, L)
        a = self._a
        _arr_r = getattr(r, "ndim", 0) != 0
        l_np = numpy.arange(0, L, dtype=float)
        l_np = l_np[:, numpy.newaxis] if _arr_r else l_np[numpy.newaxis, :]
        l = asarray_on_device(xp, l_np, device_of(r))
        rsafe = xp.where(r == 0, 1.0, r)
        generic = (
            -(a**l)
            * rsafe ** (-l - 1.0)
            / ((1.0 + a / rsafe) ** (2 * l + 1.0))
            * CC
            * (4 * numpy.pi) ** 0.5
        )
        centre = -1.0 / a * CC * (4 * numpy.pi) ** 0.5
        return xp.where(r == 0, centre, generic)

    def _compute_at_point(self, radial_func, R, z, phi, t=0.0):
        """
        Evaluate the basis-function expansion at a single point.

        Computes sum_{nlm} A_nlm * radial_nl(r) * P_l^m(cos theta) * [cos/sin](m*phi).

        Parameters
        ----------
        radial_func : function
            Radial basis function, must be _rhoTilde or _phiTilde.
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float
            Azimuth.
        t : float, optional
            Time (used only for time-dependent potentials). Default: 0.0.

        Returns
        -------
        float
            The summed density or potential at (R, z, phi).

        Notes
        -----
        - 2016-05-18 - Written - Aladdin Seaifan (UofT)
        - 2026-02-11 - Simplified - Bovy (UofT)
        """
        self._ensure_coeffs_for_time(t)
        xp = get_namespace(R, z, phi)
        N, L, M = self._Acos.shape
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        if xp is numpy:
            Acos, Asin = self._Acos, self._Asin
            # Radial part: (N, L)
            radial = radial_func(r, N, L)
            # Angular part: associated Legendre polynomials (L, M)
            PP = compute_legendre(numpy.cos(theta), L, M)
            # Azimuthal part: cos(m*phi), sin(m*phi)
            m = numpy.arange(0, M)[numpy.newaxis, numpy.newaxis, :]
            mcos = numpy.cos(m * phi)
            msin = numpy.sin(m * phi)
            return numpy.sum(
                radial[:, :, None] * (Acos * mcos + Asin * msin) * PP[None, :, :]
            )
        # backend path: same sum with the backend-agnostic special-function
        # router; the coefficient tables come from _coeffs_at_time (the fixed
        # tables when static, or the interpolated-at-t tables -- differentiable in
        # t -- when time-dependent), on the input's device (CUDA support).
        # Shape-agnostic: a SCALAR (r, theta, phi) gives the (N, L, M) sum ->
        # scalar (unchanged), while an ARRAY of shape (P,) carries a leading point
        # axis through one batched sum -> (P,), so the vectorized eval/dens path
        # needs no per-point loop. An array t makes _coeffs_at_time return
        # per-point (P, N, L, M) tables, folded into that same batched sum.
        dev = device_of(r, theta, phi)
        Acos, Asin = self._coeffs_at_time(t, xp, dev)
        radial = radial_func(r, N, L)
        PP = assoc_legendre(L, M, xp.cos(theta))
        mvec = asarray_on_device(xp, numpy.arange(0, M, dtype=float), dev)
        if getattr(r, "ndim", 0) == 0:
            m = mvec[None, None, :]
            mcos = xp.cos(m * phi)
            msin = xp.sin(m * phi)
            return xp.sum(
                radial[:, :, None] * (Acos * mcos + Asin * msin) * PP[None, :, :]
            )
        # batched: radial (N, L, P) -> (P, N, L); PP (P, L, M); azimuth (P, M);
        # contract over (N, L, M) leaving the point axis P. Acos_b/Asin_b add a
        # leading broadcast axis for scalar-t (shared) tables and pass per-point
        # (P, N, L, M) tables through unchanged for array t.
        Acos_b = Acos if getattr(Acos, "ndim", 3) == 4 else Acos[None]
        Asin_b = Asin if getattr(Asin, "ndim", 3) == 4 else Asin[None]
        radial = xp.moveaxis(radial, -1, 0)
        ang = phi[:, None] * mvec[None, :]
        mcos = xp.cos(ang)
        msin = xp.sin(ang)
        angular = Acos_b * mcos[:, None, None, :] + Asin_b * msin[:, None, None, :]
        return xp.sum(
            radial[:, :, :, None] * angular * PP[:, None, :, :], axis=(1, 2, 3)
        )

    def _evaluate_expansion(self, radial_func, R, z, phi, t=0.0):
        """
        Evaluate the basis-function expansion over an array of coordinates.

        Parameters
        ----------
        radial_func : function
            Radial basis function, must be _rhoTilde or _phiTilde.
        R : float or numpy.ndarray
            Cylindrical Galactocentric radius.
        z : float or numpy.ndarray
            Vertical height.
        phi : float or numpy.ndarray
            Azimuth.
        t : float or numpy.ndarray, optional
            Time (used only for time-dependent potentials). Default: 0.0.

        Returns
        -------
        float or numpy.ndarray
            Density or potential evaluated at (R, z, phi).

        Notes
        -----
        - 2016-06-02 - Written - Aladdin Seaifan (UofT)
        - 2026-02-11 - Simplified - Bovy (UofT)
        - 2026-07-02 - Broadcast over t for time-dependent potentials - Bovy (UofT)
        """
        xp = get_namespace(R, z, phi)
        if xp is numpy:
            R = numpy.array(R, dtype=float)
            z = numpy.array(z, dtype=float)
            phi = numpy.array(phi, dtype=float)
            t = numpy.array(t, dtype=float)
            shape = numpy.broadcast_shapes(R.shape, z.shape, phi.shape, t.shape)
            if shape == ():
                return self._compute_at_point(radial_func, R, z, phi, t=t)
            R = R * numpy.ones(shape)
            z = z * numpy.ones(shape)
            phi = phi * numpy.ones(shape)
            t = t * numpy.ones(shape)
            result = numpy.zeros(shape, float)
            for idx in numpy.ndindex(*shape):
                result[idx] = self._compute_at_point(
                    radial_func, R[idx], z[idx], phi[idx], t=t[idx]
                )
            return result
        # backend path: identical per-point evaluation, but assembled
        # functionally (stack instead of in-place writes) so it traces and
        # differentiates under jax/torch. Anchor R, z, phi on one device so a
        # CUDA array coord meeting Python-scalar siblings (which xp.asarray puts
        # on CPU) does not mix devices; dev is None for numpy -> byte-identical.
        dev = device_of(R, z, phi)
        R = asarray_on_device(xp, R, dev) * 1.0
        z = asarray_on_device(xp, z, dev) * 1.0
        phi = asarray_on_device(xp, phi, dev) * 1.0
        # broadcast t alongside the coords so a time-dependent potential picks up
        # the interpolated coefficients at each point's time (scalar t -> shared).
        t = asarray_on_device(xp, t, dev) * 1.0
        shape = (R * z * phi * t).shape
        if shape == ():
            return self._compute_at_point(radial_func, R, z, phi, t=t)
        # Vectorized: flatten the broadcast coords to 1-D and evaluate ALL points
        # in one batched _compute_at_point call (no per-point Python loop, so no
        # O(P) unrolled XLA graph / per-call retrace), then reshape back.
        R = xp.reshape(xp.broadcast_to(R, shape), (-1,))
        z = xp.reshape(xp.broadcast_to(z, shape), (-1,))
        phi = xp.reshape(xp.broadcast_to(phi, shape), (-1,))
        t = xp.reshape(xp.broadcast_to(t, shape), (-1,))
        return xp.reshape(self._compute_at_point(radial_func, R, z, phi, t=t), shape)

    def _dens(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi and phi is None:
            phi = 0.0
        # the expansion tables are deliberately float64 (precision); cast the
        # result to the input dtype at exit (backend-agnostic; no-op for
        # float64/scalar inputs)
        return match_input_dtype(
            self._evaluate_expansion(self._rhoTilde, R, z, phi, t=t), R, z, phi, t
        )

    def _mass(self, R, z=None, t=0.0):
        if not z is None:
            raise AttributeError  # Hack to fall back to general
        # when integrating over spherical volume, all non-zero l,m vanish
        xp = get_namespace(R, t)
        if xp is numpy:
            self._ensure_coeffs_for_time(t)
            N = len(self._Acos)
            return R**2.0 * numpy.sum(
                self._Acos[:, 0, 0] * self._dphiTilde(R, N, 1)[:, 0]
            )
        # backend path: the m=0,l=0 coefficient from _coeffs_at_time (fixed if
        # static, interpolated-at-t and differentiable in t if time-dependent).
        dev = device_of(R, t)
        N = len(self._Acos)
        Acos, _ = self._coeffs_at_time(t, xp, dev)
        return R**2.0 * xp.sum(Acos[:, 0, 0] * self._dphiTilde(R, N, 1)[:, 0])

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi and phi is None:
            phi = 0.0
        # float64 interior, input-dtype exit cast (see _dens)
        return match_input_dtype(
            self._evaluate_expansion(self._phiTilde, R, z, phi, t=t), R, z, phi, t
        )

    def _dphiTilde(self, r, N, L):
        xp = get_namespace(r)
        # Coerce r first (a direct call may pass a python scalar; 0.0 ** -1 would
        # raise, and _RToxi must see a backend array to route to its backend
        # path); numpy pass-through keeps the numpy branch byte-identical.
        (r,) = coerce_coords(xp, r)
        a = self._a
        xi = _RToxi(r, self._a)
        dC = _dC(xi, N, L)
        if xp is numpy:
            l = numpy.arange(0, L, dtype=float)[numpy.newaxis, :]
            n = numpy.arange(0, N, dtype=float)[:, numpy.newaxis]
            return -((4 * numpy.pi) ** 0.5) * (
                numpy.power(a * r, l)
                * (l * (a + r) * numpy.power(r, -1) - (2 * l + 1))
                / ((a + r) ** (2 * l + 2))
                * _C(xi, N, L)
                + a**-1
                * (1 - xi) ** 2
                * (a * r) ** l
                / (a + r) ** (2 * l + 1)
                * dC
                / 2.0
            )
        # backend path: identical expression with xp arithmetic. Scalar r -> l on
        # axis 1 -> (N, L); array r of shape (P,) -> l on axis 0 so the prefactor
        # is (L, P) and, against _C/dC (N, L, P), gives (N, L, P) (batched, no loop).
        _arr_r = getattr(r, "ndim", 0) != 0
        l_np = numpy.arange(0, L, dtype=float)
        l_np = l_np[:, numpy.newaxis] if _arr_r else l_np[numpy.newaxis, :]
        l = asarray_on_device(xp, l_np, device_of(r))
        return -((4 * numpy.pi) ** 0.5) * (
            (a * r) ** l
            * (l * (a + r) * r ** (-1.0) - (2 * l + 1))
            / ((a + r) ** (2 * l + 2))
            * _C(xi, N, L)
            + a**-1 * (1 - xi) ** 2 * (a * r) ** l / (a + r) ** (2 * l + 1) * dC / 2.0
        )

    def _d2phiTilde(self, r, N, L):
        # Second radial derivative of phiTilde_nl(r). phiTilde (Python
        # convention) = sqrt(4pi) x [the C-side phiTilde], so this is the C-side
        # compute_d2phiTilde expression times sqrt(4pi). C, dC, d2C are the
        # Gegenbauer polynomial and its 1st/2nd xi-derivatives; the dxi/dr
        # factors are already folded into the algebra below. (r=0 -- the centre
        # -- is a removable singularity never hit along an orbit; guarded.)
        xp = get_namespace(r)
        # Coerce r first (a direct call may pass a python/numpy scalar the xp.where
        # guards below reject, and _RToxi must see a backend array to route to its
        # backend path); numpy pass-through keeps the numpy branch byte-identical.
        (r,) = coerce_coords(xp, r)
        a = self._a
        xi = _RToxi(r, a)
        CC = _C(xi, N, L)
        dCC = _dC(xi, N, L)
        d2CC = _d2C(xi, N, L)
        if xp is numpy:
            ar = a + r
            l = numpy.arange(0, L, dtype=float)[numpy.newaxis, :]
            if r == 0:
                return numpy.zeros((N, L), float)
            ar2 = ar * ar
            ar3 = ar2 * ar
            ar4 = ar3 * ar
            # Factored as (a r / ar^2)^l / (r^2 ar^5) -- a small, stable number
            # for large r -- rather than (a r)^l / ar^(5+2l), whose intermediate
            # powers overflow to inf/inf=NaN at large l (matches the C
            # compute_d2phiTilde).
            rterm = numpy.power(a * r / ar2, l) / (r * r * ar3 * ar2)
            out = rterm * (
                CC
                * (
                    l * (1.0 - l) * ar4
                    - (4.0 * l**2 + 6.0 * l + 2.0) * r * r * ar2
                    + l * (4.0 * l + 2.0) * r * ar3
                )
                + a
                * r
                * (
                    (
                        4.0 * r * r
                        + 4.0 * a * r
                        + (8.0 * l + 4.0) * r * ar
                        - 4.0 * l * ar2
                    )
                    * dCC
                    - 4.0 * a * r * d2CC
                )
            )
            return out * (4.0 * numpy.pi) ** 0.5
        # backend path: same factored expression at a guarded radius (the r == 0
        # column is exactly zero; both xp.where branches are evaluated under
        # tracing/eager AD, so the generic branch must stay finite there). Scalar
        # r -> l on axis 1 -> (N, L); array r of shape (P,) -> l on axis 0 so the
        # r-prefactor is (L, P) and, against CC (N, L, P), gives (N, L, P).
        _arr_r = getattr(r, "ndim", 0) != 0
        l_np = numpy.arange(0, L, dtype=float)
        l_np = l_np[:, numpy.newaxis] if _arr_r else l_np[numpy.newaxis, :]
        l = asarray_on_device(xp, l_np, device_of(r))
        rs = xp.where(r == 0, 1.0, r)
        ar = a + rs
        ar2 = ar * ar
        ar3 = ar2 * ar
        ar4 = ar3 * ar
        rterm = (a * rs / ar2) ** l / (rs * rs * ar3 * ar2)
        out = rterm * (
            CC
            * (
                l * (1.0 - l) * ar4
                - (4.0 * l**2 + 6.0 * l + 2.0) * rs * rs * ar2
                + l * (4.0 * l + 2.0) * rs * ar3
            )
            + a
            * rs
            * (
                (
                    4.0 * rs * rs
                    + 4.0 * a * rs
                    + (8.0 * l + 4.0) * rs * ar
                    - 4.0 * l * ar2
                )
                * dCC
                - 4.0 * a * rs * d2CC
            )
        )
        return xp.where(r == 0, 0.0, out * (4.0 * numpy.pi) ** 0.5)

    def _compute_spher_forces_at_point(self, R, z, phi=0, t=0):
        """
        Compute spherical force components dPhi/dr, dPhi/dtheta, dPhi/dphi at a single point.

        Uses the same angular basis functions (Legendre polynomials, cos/sin(m*phi))
        as _compute_at_point, but also requires derivatives of both the radial and
        angular parts.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float
            Azimuth.

        Returns
        -------
        dPhi_dr : float
            Derivative of the potential with respect to r.
        dPhi_dtheta : float
            Derivative of the potential with respect to theta.
        dPhi_dphi : float
            Derivative of the potential with respect to phi.

        Notes
        -----
        - 2016-05-18 - Written - Aladdin Seaifan (UofT)
        - 2026-02-11 - Simplified - Bovy (UofT)
        """
        self._ensure_coeffs_for_time(t)
        xp = get_namespace(R, z, phi)
        N, L, M = self._Acos.shape
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        if xp is numpy:
            Acos, Asin = self._Acos, self._Asin
            new_hash = hashlib.md5(numpy.array([R, z, phi, t])).hexdigest()

            if new_hash == self._force_hash:
                dPhi_dr = self._cached_dPhi_dr
                dPhi_dtheta = self._cached_dPhi_dtheta
                dPhi_dphi = self._cached_dPhi_dphi
            else:
                # Angular part: Legendre polynomials and their derivatives
                PP, dPP = compute_legendre(numpy.cos(theta), L, M, deriv=True)
                PP = PP[None, :, :]
                dPP = dPP[None, :, :]
                # Radial part: potential basis and its radial derivative
                phi_tilde = self._phiTilde(r, N, L)[:, :, numpy.newaxis]
                dphi_tilde = self._dphiTilde(r, N, L)[:, :, numpy.newaxis]
                # Azimuthal part
                m = numpy.arange(0, M)[numpy.newaxis, numpy.newaxis, :]
                mcos = numpy.cos(m * phi)
                msin = numpy.sin(m * phi)
                # Coefficient-weighted angular factors
                cos_sin_sum = Acos * mcos + Asin * msin
                # Force components in spherical coordinates
                dPhi_dr = -numpy.sum(cos_sin_sum * PP * dphi_tilde)
                dPhi_dtheta = -numpy.sum(
                    cos_sin_sum * phi_tilde * dPP * (-numpy.sin(theta))
                )
                dPhi_dphi = -numpy.sum(m * (Asin * mcos - Acos * msin) * phi_tilde * PP)
                # Cache for reuse (e.g., _Rforce and _zforce called at same point)
                self._force_hash = new_hash
                self._cached_dPhi_dr = dPhi_dr
                self._cached_dPhi_dtheta = dPhi_dtheta
                self._cached_dPhi_dphi = dPhi_dphi
            return dPhi_dr, dPhi_dtheta, dPhi_dphi
        # backend path: same computation, but functional and cache-free (the
        # per-point Python hash cache is trace-hostile under jit and useless on
        # traced values; numpy keeps it above). The coefficient tables come from
        # _coeffs_at_time (fixed if static, interpolated-at-t and differentiable
        # in t if time-dependent), on the input's device (CUDA support).
        dev = device_of(r, theta, phi)
        Acos, Asin = self._coeffs_at_time(t, xp, dev)
        if getattr(r, "ndim", 0) == 0:
            # scalar: the (N, L, M) sums collapse to three scalars (unchanged).
            PP, dPP = assoc_legendre(L, M, xp.cos(theta), deriv=1)
            PP = PP[None, :, :]
            dPP = dPP[None, :, :]
            phi_tilde = self._phiTilde(r, N, L)[:, :, None]
            dphi_tilde = self._dphiTilde(r, N, L)[:, :, None]
            m = asarray_on_device(
                xp, numpy.arange(0, M, dtype=float)[None, None, :], dev
            )
            mcos = xp.cos(m * phi)
            msin = xp.sin(m * phi)
            cos_sin_sum = Acos * mcos + Asin * msin
            dPhi_dr = -xp.sum(cos_sin_sum * PP * dphi_tilde)
            dPhi_dtheta = -xp.sum(cos_sin_sum * phi_tilde * dPP * (-xp.sin(theta)))
            dPhi_dphi = -xp.sum(m * (Asin * mcos - Acos * msin) * phi_tilde * PP)
            return dPhi_dr, dPhi_dtheta, dPhi_dphi
        # batched array (P,): carry a leading point axis through the same sums.
        # Acos_b/Asin_b add a leading broadcast axis for a scalar-t (shared) table
        # or pass a per-point (P, N, L, M) table (array t) through unchanged.
        Acos_b = Acos if getattr(Acos, "ndim", 3) == 4 else Acos[None]
        Asin_b = Asin if getattr(Asin, "ndim", 3) == 4 else Asin[None]
        PP, dPP = assoc_legendre(L, M, xp.cos(theta), deriv=1)  # (P, L, M)
        PP = PP[:, None, :, :]  # (P, 1, L, M)
        dPP = dPP[:, None, :, :]
        phi_tilde = xp.moveaxis(self._phiTilde(r, N, L), -1, 0)[
            :, :, :, None
        ]  # (P,N,L,1)
        dphi_tilde = xp.moveaxis(self._dphiTilde(r, N, L), -1, 0)[:, :, :, None]
        mvec = asarray_on_device(xp, numpy.arange(0, M, dtype=float), dev)
        ang = phi[:, None] * mvec[None, :]  # (P, M)
        mcos = xp.cos(ang)[:, None, None, :]  # (P, 1, 1, M)
        msin = xp.sin(ang)[:, None, None, :]
        cos_sin_sum = Acos_b * mcos + Asin_b * msin  # (P, N, L, M)
        sin_t = xp.sin(theta)[:, None, None, None]  # (P, 1, 1, 1)
        m4 = mvec[None, None, None, :]  # (1, 1, 1, M)
        dPhi_dr = -xp.sum(cos_sin_sum * PP * dphi_tilde, axis=(1, 2, 3))
        dPhi_dtheta = -xp.sum(cos_sin_sum * phi_tilde * dPP * (-sin_t), axis=(1, 2, 3))
        dPhi_dphi = -xp.sum(
            m4 * (Asin_b * mcos - Acos_b * msin) * phi_tilde * PP,
            axis=(1, 2, 3),
        )
        return dPhi_dr, dPhi_dtheta, dPhi_dphi

    def _compute_spher_2nd_derivs_at_point(self, R, z, phi, t=0.0):
        """
        Compute the spherical-coordinate second derivatives of the potential at a
        single point: (d2Phi/dr2, d2Phi/dtheta2, d2Phi/dphi2, d2Phi/drdtheta,
        d2Phi/drdphi, d2Phi/dthetadphi, dPhi/dr, dPhi/dtheta). Fed to the
        SphericalHarmonicPotentialMixin chain-rule transform to build the
        cylindrical Hessian. The angular theta-derivatives come from the
        x=cos(theta) Legendre derivatives via dP/dtheta=-sin(theta)dP/dx and
        d2P/dtheta2=sin^2(theta)d2P/dx2-cos(theta)dP/dx.

        Notes
        -----
        - 2026-06-08 - Written - Bovy (UofT)
        """
        self._ensure_coeffs_for_time(t)
        xp = get_namespace(R, z, phi)
        N, L, M = self._Acos.shape
        if xp is numpy:
            # Cache the full spherical 2nd-derivative set: the six cylindrical
            # second derivatives at a point all transform from it, so this is
            # computed once per point instead of once per component.
            cache_key = (float(R), float(z), float(phi), float(t))
            if cache_key == self._2nd_deriv_cache_key:
                return self._cached_2nd_derivs
            Acos, Asin = self._Acos, self._Asin
            r, theta, phi = coords.cyl_to_spher(R, z, phi)
            if r == 0.0 or not numpy.isfinite(r):
                self._2nd_deriv_cache_key = cache_key
                self._cached_2nd_derivs = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                return self._cached_2nd_derivs
            costheta = numpy.cos(theta)
            sintheta = numpy.sin(theta)
            # nudge off the pole so the m>0 angular derivatives stay finite
            if M > 1 and abs(1.0 - costheta * costheta) < 1e-14:
                costheta = numpy.sign(costheta) * (1.0 - 1e-7)
                sintheta = numpy.sqrt(1.0 - costheta**2)
            PP, dPP, d2PP = compute_legendre(costheta, L, M, deriv=2)
            # theta-derivatives of the (angular) Legendre functions
            dPth = dPP * (-sintheta)
            d2Pth = d2PP * sintheta**2 - dPP * costheta
            PP = PP[numpy.newaxis, :, :]
            dPth = dPth[numpy.newaxis, :, :]
            d2Pth = d2Pth[numpy.newaxis, :, :]
            # radial basis and its first/second radial derivatives
            phi_tilde = self._phiTilde(r, N, L)[:, :, numpy.newaxis]
            dphi_tilde = self._dphiTilde(r, N, L)[:, :, numpy.newaxis]
            d2phi_tilde = self._d2phiTilde(r, N, L)[:, :, numpy.newaxis]
            m = numpy.arange(0, M)[numpy.newaxis, numpy.newaxis, :]
            mcos = numpy.cos(m * phi)
            msin = numpy.sin(m * phi)
            cos_sin = Acos * mcos + Asin * msin  # angular azimuthal coefficient
            dphi_coef = m * (Asin * mcos - Acos * msin)  # d/dphi of cos_sin
            Phi_rr = numpy.sum(cos_sin * PP * d2phi_tilde)
            Phi_tt = numpy.sum(cos_sin * d2Pth * phi_tilde)
            Phi_pp = numpy.sum(-m * m * cos_sin * PP * phi_tilde)
            Phi_rt = numpy.sum(cos_sin * dPth * dphi_tilde)
            Phi_rp = numpy.sum(dphi_coef * PP * dphi_tilde)
            Phi_tp = numpy.sum(dphi_coef * dPth * phi_tilde)
            Phi_r = numpy.sum(cos_sin * PP * dphi_tilde)
            Phi_t = numpy.sum(cos_sin * dPth * phi_tilde)
            self._2nd_deriv_cache_key = cache_key
            self._cached_2nd_derivs = (
                Phi_rr,
                Phi_tt,
                Phi_pp,
                Phi_rt,
                Phi_rp,
                Phi_tp,
                Phi_r,
                Phi_t,
            )
            return self._cached_2nd_derivs
        # backend path: cache-free (the per-point Python cache key calls
        # float(R), which is trace-hostile under jit) and branchless. The
        # r == 0 / r == inf centre is handled by computing the generic branch
        # at a guarded radius and zeroing the result with xp.where, so both
        # branches stay finite under tracing/eager AD. The coefficient tables
        # come from _coeffs_at_time (fixed if static, interpolated-at-t and
        # differentiable in t if time-dependent), on the input's device.
        # Coerce inputs: a direct call may pass python/numpy scalars that the
        # xp.isfinite / xp.where degenerate-point guards below reject.
        R, z, phi = coerce_coords(xp, R, z, phi)
        dev = device_of(R, z, phi)
        Acos, Asin = self._coeffs_at_time(t, xp, dev)
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        degenerate = (r == 0.0) | ~xp.isfinite(r)
        rs = xp.where(degenerate, 1.0, r)
        costheta = xp.cos(theta)
        sintheta = xp.sin(theta)
        if M > 1:
            # nudge off the pole so the m>0 angular derivatives stay finite
            pole = xp.abs(1.0 - costheta * costheta) < 1e-14
            costheta = xp.where(pole, xp.sign(costheta) * (1.0 - 1e-7), costheta)
            sintheta = xp.where(pole, xp.sqrt(1.0 - costheta**2), sintheta)
        if getattr(r, "ndim", 0) == 0:
            # scalar: the (N, L, M) sums collapse to eight scalars (unchanged).
            PP, dPP, d2PP = assoc_legendre(L, M, costheta, deriv=2)
            dPth = dPP * (-sintheta)
            d2Pth = d2PP * sintheta**2 - dPP * costheta
            PP = PP[None, :, :]
            dPth = dPth[None, :, :]
            d2Pth = d2Pth[None, :, :]
            phi_tilde = self._phiTilde(rs, N, L)[:, :, None]
            dphi_tilde = self._dphiTilde(rs, N, L)[:, :, None]
            d2phi_tilde = self._d2phiTilde(rs, N, L)[:, :, None]
            m = asarray_on_device(
                xp, numpy.arange(0, M, dtype=float)[None, None, :], dev
            )
            mcos = xp.cos(m * phi)
            msin = xp.sin(m * phi)
            cos_sin = Acos * mcos + Asin * msin
            dphi_coef = m * (Asin * mcos - Acos * msin)
            return tuple(
                xp.where(degenerate, 0.0, val)
                for val in (
                    xp.sum(cos_sin * PP * d2phi_tilde),  # Phi_rr
                    xp.sum(cos_sin * d2Pth * phi_tilde),  # Phi_tt
                    xp.sum(-m * m * cos_sin * PP * phi_tilde),  # Phi_pp
                    xp.sum(cos_sin * dPth * dphi_tilde),  # Phi_rt
                    xp.sum(dphi_coef * PP * dphi_tilde),  # Phi_rp
                    xp.sum(dphi_coef * dPth * phi_tilde),  # Phi_tp
                    xp.sum(cos_sin * PP * dphi_tilde),  # Phi_r
                    xp.sum(cos_sin * dPth * phi_tilde),  # Phi_t
                )
            )
        # batched array (P,): carry a leading point axis through the same sums;
        # each of the eight outputs is (P,), zeroed where the radius is degenerate.
        # Acos_b/Asin_b add a leading broadcast axis for a scalar-t (shared) table
        # or pass a per-point (P, N, L, M) table (array t) through unchanged.
        Acos_b = Acos if getattr(Acos, "ndim", 3) == 4 else Acos[None]
        Asin_b = Asin if getattr(Asin, "ndim", 3) == 4 else Asin[None]
        PP, dPP, d2PP = assoc_legendre(L, M, costheta, deriv=2)  # (P, L, M)
        st = sintheta[:, None, None]
        ct = costheta[:, None, None]
        dPth = (dPP * (-st))[:, None, :, :]  # (P, 1, L, M)
        d2Pth = (d2PP * st**2 - dPP * ct)[:, None, :, :]
        PP = PP[:, None, :, :]
        phi_tilde = xp.moveaxis(self._phiTilde(rs, N, L), -1, 0)[
            :, :, :, None
        ]  # (P,N,L,1)
        dphi_tilde = xp.moveaxis(self._dphiTilde(rs, N, L), -1, 0)[:, :, :, None]
        d2phi_tilde = xp.moveaxis(self._d2phiTilde(rs, N, L), -1, 0)[:, :, :, None]
        mvec = asarray_on_device(xp, numpy.arange(0, M, dtype=float), dev)
        ang = phi[:, None] * mvec[None, :]  # (P, M)
        mcos = xp.cos(ang)[:, None, None, :]  # (P, 1, 1, M)
        msin = xp.sin(ang)[:, None, None, :]
        cos_sin = Acos_b * mcos + Asin_b * msin  # (P, N, L, M)
        m4 = mvec[None, None, None, :]  # (1, 1, 1, M)
        dphi_coef = m4 * (Asin_b * mcos - Acos_b * msin)
        deg = degenerate
        ax = (1, 2, 3)
        return tuple(
            xp.where(deg, 0.0, val)
            for val in (
                xp.sum(cos_sin * PP * d2phi_tilde, axis=ax),  # Phi_rr
                xp.sum(cos_sin * d2Pth * phi_tilde, axis=ax),  # Phi_tt
                xp.sum(-m4 * m4 * cos_sin * PP * phi_tilde, axis=ax),  # Phi_pp
                xp.sum(cos_sin * dPth * dphi_tilde, axis=ax),  # Phi_rt
                xp.sum(dphi_coef * PP * dphi_tilde, axis=ax),  # Phi_rp
                xp.sum(dphi_coef * dPth * phi_tilde, axis=ax),  # Phi_tp
                xp.sum(cos_sin * PP * dphi_tilde, axis=ax),  # Phi_r
                xp.sum(cos_sin * dPth * phi_tilde, axis=ax),  # Phi_t
            )
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("R2", R, z, phi, t=t)

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("z2", R, z, phi, t=t)

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("Rz", R, z, phi, t=t)

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("phi2", R, z, phi, t=t)

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("Rphi", R, z, phi, t=t)

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        return self._evaluate_cyl_2nd_deriv("phiz", R, z, phi, t=t)

    def OmegaP(self):
        return 0


def _rhoTilde_basis(r, N, L, a):
    """Evaluate the SCF density basis functions rho_tilde_nl(r), shape (N, L).

    Module-level implementation of ``SCFPotential._rhoTilde`` (which depends only
    on the scale length ``a``), so the basis can be evaluated without an instance
    (e.g. when translating a MultipoleExpansionPotential into an SCFPotential).
    """
    xi = _RToxi(r, a)
    CC = _C(xi, N, L)
    rho = numpy.zeros((N, L), float)
    n = numpy.arange(0, N, dtype=float)[:, numpy.newaxis]
    l = numpy.arange(0, L, dtype=float)[numpy.newaxis, :]
    K = 0.5 * n * (n + 4 * l + 3) + (l + 1.0) * (2 * l + 1)
    rho[:, :] = (
        K
        * ((a * r) ** l)
        / ((r / a) * (a + r) ** (2 * l + 3.0))
        * CC[:, :]
        * (numpy.pi) ** -0.5
    )
    return rho


def _phiTilde_basis(r, N, L, a):
    """Evaluate the SCF potential basis functions phi_tilde_nl(r), shape (N, L).

    Module-level implementation of ``SCFPotential._phiTilde`` (see
    ``_rhoTilde_basis``).
    """
    xi = _RToxi(r, a)
    CC = _C(xi, N, L)
    phi = numpy.zeros((N, L), float)
    n = numpy.arange(0, N)[:, numpy.newaxis]
    l = numpy.arange(0, L)[numpy.newaxis, :]
    if r == 0:
        phi[:, :] = -1.0 / a * CC[:, :] * (4 * numpy.pi) ** 0.5
    else:
        phi[:, :] = (
            -(a**l)
            * r ** (-l - 1.0)
            / ((1.0 + a / r) ** (2 * l + 1.0))
            * CC[:, :]
            * (4 * numpy.pi) ** 0.5
        )
    return phi


def _interp_ppoly_vec(xp, x, c, t, dev):
    """Evaluate a scipy ``PPoly``/``CubicSpline`` with a trailing coefficient axis
    at ``t`` through the namespace ``xp``, differentiably in ``t``.

    ``x`` are the knots (shape ``(Nt,)``) and ``c`` the power-basis coefficients
    (shape ``(k+1, Nt-1, ncoeff)``, descending degree, exactly ``CubicSpline.c``):
    on ``x[i] <= t < x[i+1]`` the value is ``sum_j c[j, i] * (t - x[i])**(k-j)``.
    A ``searchsorted`` interval lookup plus Horner over the (static) polynomial
    degree; the interval index is clamped to ``[0, Nt-2]`` so a ``t`` outside the
    grid evaluates the edge polynomial (finite extrapolation, matching scipy's
    default ``extrapolate=True`` and byte-for-byte-equivalent to the numpy path to
    ~1 ulp). This mirrors ``galpy.backend.interpolate.eval_ppoly`` but broadcasts
    the ``(t - x)`` Horner factor against the extra trailing ``ncoeff`` axis so a
    ``(P,)`` time array yields ``(P, ncoeff)`` (each point its own time). Returns
    ``(ncoeff,)`` for a scalar ``t`` or ``(P, ncoeff)`` for a ``(P,)`` array ``t``.
    """
    xb = asarray_on_device(xp, numpy.asarray(x), dev)
    cb = asarray_on_device(xp, numpy.asarray(c), dev)
    idx = xp.clip(xp.searchsorted(xb, t, side="right") - 1, 0, cb.shape[1] - 1)
    dt = t - xb[idx]
    if getattr(t, "ndim", 0) != 0:
        dt = dt[:, None]  # (P, 1): broadcast Horner over the trailing coeff axis
    out = cb[0, idx]
    for j in range(1, cb.shape[0]):
        out = out * dt + cb[j, idx]
    return out


def _xiToR(xi, a=1):
    # Namespace-dispatched like _RToxi: a bare numpy.divide on a backend array
    # returns numpy and SILENTLY drops the gradient, which would sever the
    # coefficient quadrature from the density parameters it is differentiated
    # against. numpy input takes the identical numpy.divide call as before.
    xp = get_namespace(xi)
    if xp is numpy:
        return a * numpy.divide((1.0 + xi), (1.0 - xi))
    return a * ((1.0 + xi) / (1.0 - xi))


def _RToxi(r, a=1):
    # Leaf helper consumed by both the numpy coefficient setup and the backend
    # evaluation: dispatch on the DATA, not the (possibly forced) namespace, so a
    # numpy/scalar r stays numpy (byte-identical, keeps numpy parents working)
    # while a backend array routes to the differentiable backend path.
    if not is_backend_array(r):
        out = numpy.divide((r / a - 1.0), (r / a + 1.0), where=True ^ numpy.isinf(r))
        if numpy.any(numpy.isinf(r)):
            if hasattr(r, "__len__"):
                out[numpy.isinf(r)] = 1.0
            else:
                return 1.0
        return out
    # backend path: functional version of the above. The division is computed
    # at a guarded radius (inf/inf = NaN would poison tracing/eager AD) and the
    # r = inf entries are set to the xi = 1 limit with xp.where.
    xp = get_namespace(r)
    rsafe = xp.where(xp.isinf(r), 0.0, r)
    return xp.where(xp.isinf(r), 1.0, (rsafe / a - 1.0) / (rsafe / a + 1.0))


def _coeff_dens_numpy(val):
    """Cast a coefficient-quadrature density value to numpy.

    The coefficient setup runs under a forced-numpy context, but a user density
    that closes over a backend-amp potential still returns a backend array; cast
    it to numpy so the numpy quadrature works. No-op (byte-identical) for
    numpy/Quantity outputs, which are not backend arrays.
    """
    return as_numpy(val) if is_backend_array(val) else val


def _C(xi, N, L, alpha=lambda x: 2 * x + 3.0 / 2, singleL=False):
    """
    Evaluate the Gegenbauer polynomial for 0 <= l < L and 0<= n < N

    Parameters
    ----------
    xi : float
        Radial transformed variable
    N : int
        Size of the N dimension
    L : int
        Size of the L dimension
    alpha : function, optional
        A lambda function of l. Default alpha = 2l + 3/2
    singleL : bool, optional
        If True only compute the L-th polynomial (default: False)

    Returns
    -------
    numpy.ndarray
        An LxN Gegenbauer Polynomial

    Notes
    -----
    - 2016-05-16 - Written - Aladdin Seaifan (UofT)
    - 2021-02-22 - Upgraded to array xi - Bovy (UofT)
    - 2021-02-22 - Added singleL for use in compute...nbody - Bovy (UofT)
    """
    xp = get_namespace(xi)
    if xp is numpy:
        floatIn = False
        if isinstance(xi, (float, int)):
            floatIn = True
            xi = numpy.array([xi])
        if singleL:
            Ls = [L]
        else:
            Ls = range(L)
        CC = numpy.zeros((N, len(Ls), len(xi)))
        for l, ll in enumerate(Ls):
            for n in range(N):
                a = alpha(ll)
                if n == 0:
                    CC[n, l] = 1.0
                    continue
                elif n == 1:
                    CC[n, l] = 2.0 * a * xi
                if n + 1 != N:
                    CC[n + 1, l] = (
                        2 * (n + a) * xi * CC[n, l] - (n + 2 * a - 1) * CC[n - 1, l]
                    ) / (n + 1.0)
        if floatIn:
            return CC[:, :, 0]
        else:
            return CC
    # backend path: the special-function router's Gegenbauer recurrence (same
    # three-term recurrence, built functionally so it traces/differentiates).
    # gegenbauer returns xi.shape + (N,); stack the l values along the last
    # axis and move (n, l) to the front to match the numpy layout: (N, len(Ls))
    # for scalar xi, (N, len(Ls)) + xi.shape for array xi.
    Ls = [L] if singleL else range(L)
    CC = xp.stack([gegenbauer(N, alpha(ll), xi) for ll in Ls], axis=-1)
    return xp.moveaxis(CC, (-2, -1), (0, 1))


def _dC(xi, N, L):
    xp = get_namespace(xi)
    l = numpy.arange(0, L)[numpy.newaxis, :]
    CC = _C(xi, N + 1, L, alpha=lambda x: 2 * x + 5.0 / 2)
    if xp is numpy:
        CC = numpy.roll(CC, 1, axis=0)[:-1, :]
        CC[0, :] = 0
        CC *= 2 * (2 * l + 3.0 / 2)
        return CC
    # backend path: the roll + zero-row idiom above (dC_n = 2 a C_{n-1}^{a+1},
    # with a zero n=0 row), written functionally as a concatenation. For an array
    # xi of shape (P,) (the batched force path) _C returns (N, L, P), so the
    # (1, L) l-factor gains a trailing point axis to broadcast.
    CC = xp.concat([xp.zeros_like(CC[:1]), CC[: N - 1]], axis=0)
    fac = 2 * (2 * l + 3.0 / 2)
    if getattr(xi, "ndim", 0) != 0:
        fac = fac[:, :, numpy.newaxis]
    return CC * asarray_on_device(xp, fac, device_of(xi))


def _d2C(xi, N, L):
    # Second xi-derivative of the Gegenbauer polynomials, via
    #   d2C_n^a/dxi2 = 4 a (a+1) C_{n-2}^{a+2}(xi),   a = 2l + 3/2
    # (the analogue of _dC's dC_n^a/dxi = 2a C_{n-1}^{a+1}).
    xp = get_namespace(xi)
    l = numpy.arange(0, L)[numpy.newaxis, :]
    a = 2 * l + 3.0 / 2
    CC = _C(xi, N + 2, L, alpha=lambda x: 2 * x + 7.0 / 2)
    if xp is numpy:
        CC = numpy.roll(CC, 2, axis=0)[:-2, :]
        # n=0 (and n=1, when present) have zero second derivative; for N=1 only
        # the n=0 row exists, so guard the n=1 assignment.
        CC[0, :] = 0
        if N > 1:
            CC[1, :] = 0
        CC *= 4 * a * (a + 1)
        return CC
    # backend path: the roll + zero-rows idiom above, written functionally as a
    # concatenation (min(N, 2) zero rows, then C_{n-2}^{a+2} for n >= 2). For an
    # array xi of shape (P,) _C returns (N, L, P), so the (1, L) a-factor gains a
    # trailing point axis to broadcast.
    CC = xp.concat([xp.zeros_like(CC[: min(N, 2)]), CC[: max(N - 2, 0)]], axis=0)
    fac = 4 * a * (a + 1)
    if getattr(xi, "ndim", 0) != 0:
        fac = fac[:, :, numpy.newaxis]
    return CC * asarray_on_device(xp, fac, device_of(xi))


def scf_compute_coeffs_spherical_nbody(pos, N, mass=1.0, a=1.0):
    """
    Numerically compute the expansion coefficients for a spherical expansion for a given $N$-body set of points

    Parameters
    ----------
    pos : numpy.ndarray
        Positions of particles in rectangular coordinates with shape [3,n]
    N : int
        Size of the Nth dimension of the expansion coefficients
    mass : float or numpy.ndarray, optional
        Mass of particles (scalar or array with size n), by default 1.0
    a : float, optional
        Parameter used to scale the radius, by default 1.0

    Returns
    -------
    tuple
        Expansion coefficients for density dens that can be given to SCFPotential.__init__

    Notes
    -----
    - 2020-11-18 - Written - Morgan Bennett (UofT)
    - 2021-02-22 - Sped-up - Bovy (UofT)

    """
    # Construction-time numerical setup: pin to numpy so the particle-sum basis
    # (via the namespace-dispatched _RToxi/_C) runs on numpy regardless of any
    # forced backend default (byte-identical no-op on the numpy backend).
    # Follows the ambient namespace so backend particle positions/masses give
    # coefficients differentiable w.r.t. them. einsum is used (not a
    # sum-of-products rewrite) because its summation ORDER is what keeps the
    # numpy result byte-identical: the obvious alternatives differ in the last
    # bits (3.6e-15 for (B*A).sum(-1), 7.1e-15 for B @ A).
    _xp = get_namespace(pos)
    # get_namespace follows the AMBIENT namespace, so under a forced backend it
    # returns e.g. torch even for numpy positions; carry them across or the very
    # first _xp call raises. No-op on numpy.
    pos = _xp.asarray(pos)
    Asin = None
    r = _xp.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
    RhoSum = _xp.einsum("j,ij", mass / (1.0 + r / a), _C(_RToxi(r, a=a), N, 1)[:, 0])
    n = numpy.arange(0, N)
    K = 4 * (n + 3.0 / 2) / ((n + 2) * (n + 1) * (1 + n * (n + 3.0) / 2.0))
    Acos = get_namespace(RhoSum).reshape(RhoSum * like(RhoSum, 2 * K), (N, 1, 1))
    return Acos, Asin


def _scf_compute_determine_dens_kwargs(dens, param):
    try:
        param[0] = 1.0
        dens(*param, use_physical=False)
    except:
        dens_kw = {}
    else:
        dens_kw = {"use_physical": False}
    return dens_kw


def scf_compute_coeffs_spherical(dens, N, a=1.0, radial_order=None):
    """
    Numerically compute the expansion coefficients for a given spherical density

    Parameters
    ----------
    dens : function
        A density function that takes a parameter R
    N : int
        Size of expansion coefficients
    a : float, optional
        Parameter used to scale the radius (default is 1.0)
    radial_order : int, optional
        Number of sample points of the radial integral. If None, radial_order=max(20, N + 1) (default is None)

    Returns
    -------
    tuple
        (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

    Notes
    -----
    - 2016-05-18 - Written - Aladdin Seaifan (UofT)
    """
    # The density-arity autodetect PROBES user code with try/except, so it stays
    # pinned to numpy: a forced backend can change which exception a wrong-arity
    # call raises, and the probe must not depend on that. The quadrature below is
    # deliberately NOT pinned -- it follows the ambient namespace, so a backend
    # density yields coefficients differentiable w.r.t. its parameters.
    with _use_backend("numpy", force=True):
        numOfParam = 0
        try:
            dens(0)
            numOfParam = 1
        except:
            try:
                dens(0, 0)
                numOfParam = 2
            except:
                numOfParam = 3
        param = [0] * numOfParam
        dens_kw = _scf_compute_determine_dens_kwargs(dens, param)

    def integrand(xi):
        r = _xiToR(xi, a)
        R = r
        param[0] = R
        return (
            a**3.0
            * dens(*param, **dens_kw)
            * (1 + xi) ** 2.0
            * (1 - xi) ** -3.0
            * _C(xi, N, 1)[:, 0]
        )

    Asin = None

    Ksample = [max(N + 1, 20)]

    if radial_order != None:
        Ksample[0] = radial_order

    integrated = _gaussianQuadrature(integrand, [[-1.0, 1.0]], Ksample=Ksample)
    n = numpy.arange(0, N)
    K = 16 * numpy.pi * (n + 3.0 / 2) / ((n + 2) * (n + 1) * (1 + n * (n + 3.0) / 2.0))
    # Built functionally rather than by assigning into a preallocated numpy
    # array, which would neither trace nor accept a backend value. K is carried
    # onto the result's namespace/device first (same idiom as _computeArray's
    # K above): mixing a numpy K with a torch Tensor lets numpy own the
    # operation, and numpy resolves that by calling .numpy() on the Tensor,
    # which RAISES once it requires grad.
    _xp = get_namespace(integrated)
    Acos = _xp.reshape(integrated * like(integrated, 2 * K), (N, 1, 1))
    return Acos, Asin


def scf_compute_coeffs_axi_nbody(pos, N, L, mass=1.0, a=1.0):
    """
    Numerically compute the expansion coefficients for a given $N$-body set of points assuming that the density is axisymmetric

    Parameters
    ----------
    pos : numpy.ndarray
        Positions of particles in rectangular coordinates with shape [3,n]
    N : int
        Size of the Nth dimension of the expansion coefficients
    L : int
        Size of the Lth dimension of the expansion coefficients
    mass : float or numpy.ndarray, optional
        Mass of particles (scalar or array with size n), by default 1.0
    a : float, optional
        Parameter used to scale the radius, by default 1.0

    Returns
    -------
    tuple
        Expansion coefficients for density dens that can be given to SCFPotential.__init__

    Notes
    -----
    - 2021-02-22 - Written based on general code - Bovy (UofT)
    """
    # Construction-time numerical setup: pin to numpy so the particle-sum basis
    # (via the namespace-dispatched _RToxi/_C) runs on numpy regardless of any
    # forced backend default (byte-identical no-op on the numpy backend).
    # Follows the ambient namespace; the Legendre recursion below is pure
    # arithmetic on Python names, so it carries over unchanged.
    _xp = get_namespace(pos)
    # get_namespace follows the AMBIENT namespace, so under a forced backend it
    # returns e.g. torch even for numpy positions; carry them across or the very
    # first _xp call raises. No-op on numpy.
    pos = _xp.asarray(pos)
    r = _xp.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
    costheta = pos[2] / r
    mass = _xp.asarray(mass)
    if mass.ndim == 0:  # what numpy.atleast_1d did here
        mass = _xp.reshape(mass, (1,))
    Asin = None
    _cols = []  # per-l columns, stacked at the end instead of assigning
    # into a preallocated numpy array (which neither traces nor accepts a
    # backend value). Same values in the same order, so numpy is unchanged.
    Pll = _xp.ones(len(r))  # Set up Assoc. Legendre recursion
    # (n,l) dependent constant
    n = numpy.arange(0, N)[:, numpy.newaxis]
    l = numpy.arange(0, L)[numpy.newaxis, :]
    Knl = 0.5 * n * (n + 4.0 * l + 3.0) + (l + 1) * (2.0 * l + 1.0)
    Inl = (
        -Knl
        * 2.0
        * numpy.pi
        / 2.0 ** (8.0 * l + 6.0)
        * gamma(n + 4.0 * l + 3.0)
        / gamma(n + 1)
        / (n + 2.0 * l + 1.5)
        / gamma(2.0 * l + 1.5) ** 2
        / numpy.sqrt(2.0 * l + 1)
    )
    # Set up Assoc. Legendre recursion
    Plm = Pll
    Plmm1 = 0.0
    for ll in range(L):
        # Compute Gegenbauer polys for this l
        Cn = _C(_RToxi(r, a=a), N, ll, singleL=True)
        phinlm = -((r / a) ** ll) / (1.0 + r / a) ** (2.0 * ll + 1) * Cn[:, 0] * Plm
        # Acos
        Sum = _xp.sum(mass[numpy.newaxis, :] * phinlm, axis=-1)
        _cols.append(Sum / like(Sum, Inl[:, ll]))
        # Recurse Assoc. Legendre
        if ll < L:
            tmp = Plm
            Plm = ((2 * ll + 1.0) * costheta * Plm - ll * Plmm1) / (ll + 1)
            Plmm1 = tmp
    _out = (
        _cols[0] if len(_cols) == 1 else get_namespace(_cols[0]).stack(_cols, axis=-1)
    )
    Acos = get_namespace(_out).reshape(_out, (N, L, 1))
    return Acos, Asin


def scf_compute_coeffs_axi(dens, N, L, a=1.0, radial_order=None, costheta_order=None):
    """
    Numerically compute the expansion coefficients for a given axi-symmetric density

    Parameters
    ----------
    dens : function
        A density function that takes parameters R and z
    N : int
        Size of the Nth dimension of the expansion coefficients
    L : int
        Size of the Lth dimension of the expansion coefficients
    a : float, optional
        Parameter used to shift the basis functions (default is 1.0)
    radial_order : int, optional
        Number of sample points of the radial integral. If None, radial_order=max(20, N + 3/2L + 1) (default is None)
    costheta_order : int, optional
        Number of sample points of the costheta integral. If None, If costheta_order=max(20, L + 1) (default is None)

    Returns
    -------
    tuple
        (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

    Notes
    -----
    - 2016-05-20 - Written - Aladdin Seaifan (UofT)
    """
    # Construction-time numerical setup: pin to numpy (see scf_compute_coeffs_spherical).
    # Only the density-arity autodetect stays pinned: it PROBES user code with
    # try/except, and a forced backend can change which exception a wrong-arity
    # call raises. The quadrature below follows the ambient namespace.
    with _use_backend("numpy", force=True):
        numOfParam = 0
        try:
            dens(0, 0)
            numOfParam = 2
        except:
            numOfParam = 3
        param = [0] * numOfParam
        dens_kw = _scf_compute_determine_dens_kwargs(dens, param)

    def integrand(xi, costheta):
        l = numpy.arange(0, L)[numpy.newaxis, :]
        r = _xiToR(xi, a)
        R = r * numpy.sqrt(1 - costheta**2.0)
        z = r * costheta
        # The special-function router replaces the scipy-version fork: it is
        # byte-identical to assoc_legendre_p_all(L-1, 0, ct, branch_cut=2)[0]
        # on numpy and handles the pre-1.15 lpmn spelling internally, while
        # tracing and differentiating on a backend.
        PP = assoc_legendre(L, 1, costheta)[..., 0][numpy.newaxis, :]
        dV = (1.0 + xi) ** 2.0 * numpy.power(1.0 - xi, -4.0)
        _CC = _C(xi, N, L)[:, :]
        # The (n, l) prefactor is built with numpy.arange, so on a backend it
        # would own `prefactor * _CC` and call .numpy() on it. `like` carries it
        # across (a no-op on numpy) WITHOUT reordering or regrouping the product.
        _pref = like(_CC, a**3 * (1.0 + xi) ** l * (1.0 - xi) ** (l + 1.0))
        phi_nl = _pref * _CC * PP
        param[0] = R
        param[1] = z
        return phi_nl * dV * dens(*param, **dens_kw)

    Asin = None

    ##This should save us some computation time since we're only taking the double integral once, rather then L times
    Ksample = [max(N + 3 * L // 2 + 1, 20), max(L + 1, 20)]
    if radial_order != None:
        Ksample[0] = radial_order
    if costheta_order != None:
        Ksample[1] = costheta_order

    integrated = _gaussianQuadrature(integrand, [[-1, 1], [-1, 1]], Ksample=Ksample) * (
        2 * numpy.pi
    )
    n = numpy.arange(0, N)[:, numpy.newaxis]
    l = numpy.arange(0, L)[numpy.newaxis, :]
    K = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    # I = -K*(4*numpy.pi)/(2.**(8*l + 6)) * gamma(n + 4*l + 3)/(gamma(n + 1)*(n + 2*l + 3./2)*gamma(2*l + 3./2)**2)
    ##Taking the ln of I will allow bigger size coefficients
    lnI = (
        -(8 * l + 6) * numpy.log(2)
        + gammaln(n + 4 * l + 3)
        - gammaln(n + 1)
        - numpy.log(n + 2 * l + 3.0 / 2)
        - 2 * gammaln(2 * l + 3.0 / 2)
    )
    I = -K * (4 * numpy.pi) * numpy.e ** (lnI)
    constants = -(2.0 ** (-2 * l)) * (2 * l + 1.0) ** 0.5
    # `2 * I**-1 * integrated * constants` groups as ((2*I**-1) * integrated) *
    # constants. Only the FIRST product is commuted so the backend array leads
    # (numpy otherwise owns `ndarray * Tensor` and calls .numpy() on it); the
    # grouping is preserved because float multiplication is commutative but NOT
    # associative, so regrouping would change the numpy bits.
    _xp = get_namespace(integrated)
    _fac, _con = like(integrated, 2 * I**-1, constants)
    Acos = _xp.reshape((integrated * _fac) * _con, (N, L, 1))
    return Acos, Asin


def _stack_lm_grid(grid, N, L):
    """Assemble a [ll][mm] list-of-lists of (N,) columns into (N, L, L).

    Entries left as None (the ll < mm half the recursions never fill) become
    zeros, matching the preallocated-zeros arrays this replaces.
    """
    filled = next(c for row in grid for c in row if c is not None)
    xp = get_namespace(filled)
    zero = xp.zeros_like(filled)
    rows = [
        xp.stack(
            [grid[ll][mm] if grid[ll][mm] is not None else zero for mm in range(L)],
            axis=-1,
        )
        for ll in range(L)
    ]
    return xp.stack(rows, axis=1)


def scf_compute_coeffs_nbody(pos, N, L, mass=1.0, a=1.0):
    """
    Numerically compute the expansion coefficients for a given $N$-body set of points

    Parameters
    ----------
    pos : numpy.ndarray
        Positions of particles in rectangular coordinates with shape [3,n]
    N : int
        Size of the Nth dimension of the expansion coefficients
    L : int
        Size of the Lth and Mth dimension of the expansion coefficients
    mass : float or numpy.ndarray, optional
        Mass of particles (scalar or array with size n), by default 1.0
    a : float, optional
        Parameter used to scale the radius, by default 1.0

    Returns
    -------
    tuple
        Expansion coefficients for density dens that can be given to SCFPotential.__init__

    Notes
    -----
    - 2020-11-18 - Written - Morgan Bennett (UofT)

    """
    # Follows the ambient namespace. The (l, m) grids are collected as Python
    # lists and stacked once at the end: the original assigned into preallocated
    # numpy arrays at only the ll >= mm entries, leaving the rest zero, which
    # neither traces nor accepts a backend value. Same values in the same order.
    _xp = get_namespace(pos)
    # get_namespace follows the AMBIENT namespace, so under a forced backend it
    # returns e.g. torch even for numpy positions; carry them across or the very
    # first _xp call raises. No-op on numpy.
    pos = _xp.asarray(pos)
    r = _xp.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
    phi = _xp.atan2(pos[1], pos[0])
    costheta = pos[2] / r
    sintheta = _xp.sqrt(1.0 - costheta**2.0)
    mass = _xp.asarray(mass)
    if mass.ndim == 0:  # what numpy.atleast_1d did here
        mass = _xp.reshape(mass, (1,))
    _cos = [[None] * L for _ in range(L)]  # [ll][mm]
    _sin = [[None] * L for _ in range(L)]
    Pll = _xp.ones(len(r))  # Set up Assoc. Legendre recursion
    # (n,l) dependent constant
    n = numpy.arange(0, N)[:, numpy.newaxis]
    l = numpy.arange(0, L)[numpy.newaxis, :]
    Knl = 0.5 * n * (n + 4.0 * l + 3.0) + (l + 1) * (2.0 * l + 1.0)
    Inl = (
        -Knl
        * 2.0
        * numpy.pi
        / 2.0 ** (8.0 * l + 6.0)
        * gamma(n + 4.0 * l + 3.0)
        / gamma(n + 1)
        / (n + 2.0 * l + 1.5)
        / gamma(2.0 * l + 1.5) ** 2
    )
    for mm in range(L):  # Loop over m
        cosmphi = _xp.cos(phi * mm)
        sinmphi = _xp.sin(phi * mm)
        # Set up Assoc. Legendre recursion
        Plm = Pll
        Plmm1 = 0.0
        for ll in range(mm, L):
            # Compute Gegenbauer polys for this l
            Cn = _C(_RToxi(r, a=a), N, ll, singleL=True)
            phinlm = -((r / a) ** ll) / (1.0 + r / a) ** (2.0 * ll + 1) * Cn[:, 0] * Plm
            # Acos
            Sum = numpy.sqrt(
                (2.0 * ll + 1) * gamma(ll - mm + 1) / gamma(ll + mm + 1)
            ) * _xp.sum((mass * cosmphi)[numpy.newaxis, :] * phinlm, axis=-1)
            _cos[ll][mm] = Sum / like(Sum, Inl[:, ll])
            # Asin
            Sum = numpy.sqrt(
                (2.0 * ll + 1) * gamma(ll - mm + 1) / gamma(ll + mm + 1)
            ) * _xp.sum((mass * sinmphi)[numpy.newaxis, :] * phinlm, axis=-1)
            _sin[ll][mm] = Sum / like(Sum, Inl[:, ll])
            # Recurse Assoc. Legendre
            if ll < L:
                tmp = Plm
                Plm = ((2 * ll + 1.0) * costheta * Plm - (ll + mm) * Plmm1) / (
                    ll - mm + 1
                )
                Plmm1 = tmp
        # Recurse Assoc. Legendre (out-of-place: an in-place *= on a
        # grad-tracking tensor would break the graph)
        Pll = Pll * (-(2 * mm + 1.0) * sintheta)
    Acos = _stack_lm_grid(_cos, N, L)
    Asin = _stack_lm_grid(_sin, N, L)
    return Acos, Asin


# Peak-memory budget (bytes) for one working copy of the per-particle basis
# arrays [shape (N, batch)] when computing N-body coefficients; the particle sum
# is accumulated in batches no larger than this so building from a very large
# number of particles stays memory-bounded. Module-level (not a public
# parameter); tests set it small to exercise the batched path.
_NBODY_BATCH_BYTES = 32 * 1024**2  # 32 MB


def _nbody_parse_mass(mass, n):
    """Normalize a particle-mass input (scalar or length-n array) to shape (n,).

    Notes
    -----
    - 2026-07-04 - Written - Bovy (UofT)
    """
    mass = numpy.asarray(mass, dtype=float)
    if mass.ndim == 0 or mass.size == 1:
        return numpy.broadcast_to(mass.reshape(()), (n,))
    if mass.shape != (n,):
        raise ValueError("mass must be a scalar or match the number of particles")
    return mass


def _nbody_symmetry_coeffs(pos, N, L, mass, a, symmetry):
    """Compute (Acos, Asin) from particle positions/masses for the assumed symmetry.

    Notes
    -----
    - 2026-07-04 - Written - Bovy (UofT)
    """
    if symmetry is not None and symmetry.startswith("spher"):
        return scf_compute_coeffs_spherical_nbody(pos, N, mass=mass, a=a)
    elif symmetry is not None and symmetry.startswith("axi"):
        return scf_compute_coeffs_axi_nbody(pos, N, L, mass=mass, a=a)
    else:
        return scf_compute_coeffs_nbody(pos, N, L, mass=mass, a=a)


def _nbody_batch_size(n, N):
    """Number of particles to process per batch so a working copy of the
    (N, batch) per-particle basis arrays stays within ``_NBODY_BATCH_BYTES``.

    Notes
    -----
    - 2026-07-04 - Written - Bovy (UofT)
    """
    per_batch = _NBODY_BATCH_BYTES // (max(N, 1) * 8 * 4)  # ~4 working copies
    return int(min(n, max(1, per_batch)))


def _batched_nbody(pos, N, L, mass, a, symmetry):
    """Compute (Acos, Asin) from particle positions ``pos`` [shape (3,n)] and
    masses ``mass`` [shape (n,)], accumulating the particle sum in batches to
    bound memory. This is exact: each coefficient is a particle-independent
    constant times a sum over particles, so summing per-batch coefficients
    reproduces the all-at-once result (up to floating-point summation order).

    Notes
    -----
    - 2026-07-04 - Written - Bovy (UofT)
    """
    if (symmetry is None or not symmetry.startswith("spher")) and L is None:
        raise ValueError("L must be specified unless symmetry='spherical'")
    n = pos.shape[1]
    batch = _nbody_batch_size(n, N)
    Acos = None
    Asin = None
    for start in range(0, n, batch):
        sl = slice(start, start + batch)
        Ac, As = _nbody_symmetry_coeffs(pos[:, sl], N, L, mass[sl], a, symmetry)
        Acos = Ac if Acos is None else Acos + Ac
        if As is not None:
            Asin = As if Asin is None else Asin + As
    return Acos, Asin


def _scf_coeffs_from_multipole(mult, N, L, M, a, radial_order, t=None):
    """Project a ``MultipoleExpansionPotential``'s density multipoles onto the SCF
    radial basis, returning beta-baked SCF coefficients (Acos, Asin), each of shape
    (N, L, M).

    Both expansions use the same real spherical harmonics, so the density
    multipole ``D_lm(r)`` (the coefficient of ``P_l^m(cos theta) cos/sin(m phi)``,
    which the multipole stores beta-baked) is projected onto the SCF radial basis
    ``phi_tilde_nl`` using the biorthogonality of the density/potential bases:
    ``A_nlm = (1/W_nl) int D_lm(r) phi_tilde_nl(r) r^2 dr`` with
    ``W_nl = int rho_tilde_nl phi_tilde_nl r^2 dr`` (diagonal in n). ``t`` selects
    the snapshot for a time-dependent multipole (None for a static one).

    Notes
    -----
    - 2026-07-04 - Written - Bovy (UofT)
    """
    # Construction-time numerical setup: pin to numpy so the radial-projection
    # basis (via the namespace-dispatched _rhoTilde_basis/_phiTilde_basis/_RToxi)
    # runs on numpy regardless of any forced backend default (byte-identical no-op
    # on the numpy backend).
    with _use_backend("numpy", force=True):
        rmin, rmax = mult._rgrid[0], mult._rgrid[-1]
        if t is None:  # static: use the stored density-multipole splines
            cos_splines = mult._rho_cos_splines
            sin_splines = mult._rho_sin_splines
        else:  # time-dependent: interpolate rho_lm on the multipole rgrid at t
            cos_splines = [[None] * M for _ in range(L)]
            sin_splines = [[None] * M for _ in range(L)]
            for l in range(L):
                for mm in range(min(l + 1, M)):
                    cos_splines[l][mm] = InterpolatedUnivariateSpline(
                        mult._rgrid, mult._rho_cos_interp[l][mm](t), k=3
                    )
                    if mm > 0:  # the m=0 sine coefficient is identically zero
                        sin_splines[l][mm] = InterpolatedUnivariateSpline(
                            mult._rgrid, mult._rho_sin_interp[l][mm](t), k=3
                        )

        def _eval(splines, rq):
            # density multipoles D_lm(rq) matching the multipole's extrapolation
            # (clamp below rmin, zero above rmax); shape (L, M, len(rq))
            out = numpy.zeros((L, M, len(rq)))
            rr = numpy.clip(rq, rmin, rmax)
            beyond = rq > rmax
            for l in range(L):
                for mm in range(min(l + 1, M)):
                    if splines[l][mm] is None:
                        continue
                    v = splines[l][mm](rr)
                    v[beyond] = 0.0
                    out[l, mm] = v
            return out

        K = radial_order if radial_order is not None else max(2 * N + L, 200)
        xi, w = leggauss(K)
        rq = _xiToR(xi, a)
        weight = w * (2.0 * a / (1.0 - xi) ** 2.0) * rq**2.0  # w * dr/dxi * r^2
        rhoTq = numpy.array([_rhoTilde_basis(r, N, L, a) for r in rq]).transpose(
            1, 2, 0
        )
        phiTq = numpy.array([_phiTilde_basis(r, N, L, a) for r in rq]).transpose(
            1, 2, 0
        )
        Wnl = numpy.einsum("nlk,nlk,k->nl", rhoTq, phiTq, weight)  # (N,L) diag in n
        Dcos = _eval(cos_splines, rq)
        Dsin = _eval(sin_splines, rq)
        Acos = numpy.einsum("lmk,nlk,k->nlm", Dcos, phiTq, weight) / Wnl[:, :, None]
        Asin = numpy.einsum("lmk,nlk,k->nlm", Dsin, phiTq, weight) / Wnl[:, :, None]
    return Acos, Asin


def scf_compute_coeffs(
    dens, N, L, a=1.0, radial_order=None, costheta_order=None, phi_order=None
):
    """
    Numerically compute the expansion coefficients for a given triaxial density

    Parameters
    ----------
    dens : function
        A density function that takes parameters R, z and phi
    N : int
        Size of the Nth dimension of the expansion coefficients
    L : int
        Size of the Lth and Mth dimension of the expansion coefficients
    a : float, optional
        Parameter used to shift the basis functions (default is 1.0)
    radial_order : int, optional
        Number of sample points of the radial integral. If None, radial_order=max(20, N + 3/2L + 1) (default is None)
    costheta_order : int, optional
        Number of sample points of the costheta integral. If None, If costheta_order=max(20, L + 1) (default is None)
    phi_order : int, optional
        Number of sample points of the phi integral. If None, If costheta_order=max(20, L + 1) (default is None)

    Returns
    -------
    tuple
        (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

    Notes
    -----
    - 2016-05-27 - Written - Aladdin Seaifan (UofT)

    """
    # Only the density-kwargs probe stays pinned to numpy (it PROBES user code
    # with try/except); the quadrature follows the ambient namespace.
    with _use_backend("numpy", force=True):
        dens_kw = _scf_compute_determine_dens_kwargs(dens, [0.1, 0.1, 0.1])

    def integrand(xi, costheta, phi):
        l = numpy.arange(0, L)[numpy.newaxis, :, numpy.newaxis]
        m = numpy.arange(0, L)[numpy.newaxis, numpy.newaxis, :]
        r = _xiToR(xi, a)
        R = r * numpy.sqrt(1 - costheta**2.0)
        z = r * costheta
        # Router call, byte-identical on numpy to the scipy expression it
        # replaces (swapaxes+.T cancel for 2-D), and traceable on a backend.
        PP = assoc_legendre(L, L, costheta)[numpy.newaxis, :, :]
        dV = (1.0 + xi) ** 2.0 * numpy.power(1.0 - xi, -4.0)

        _CC = _C(xi, N, L)[:, :, numpy.newaxis]
        _pref = like(_CC, -(a**3) * (1.0 + xi) ** l * (1.0 - xi) ** (l + 1.0))
        phi_nl = _pref * _CC * PP

        _dens = dens(R, z, phi, **dens_kw)
        _cs = numpy.array([numpy.cos(m * phi), numpy.sin(m * phi)])
        # _cs is a numpy ARRAY, so it would own `backend * _cs` and resolve it
        # by calling .numpy() on a grad-tracking tensor; anchor it first. The
        # left-to-right grouping is untouched, so numpy stays byte-identical.
        _cs = like(phi_nl, _cs)
        return _dens * phi_nl[numpy.newaxis, :, :, :] * _cs * dV

    Ksample = [max(N + 3 * L // 2 + 1, 20), max(L + 1, 20), max(L + 1, 20)]
    if radial_order != None:
        Ksample[0] = radial_order
    if costheta_order != None:
        Ksample[1] = costheta_order
    if phi_order != None:
        Ksample[2] = phi_order
    integrated = _gaussianQuadrature(
        integrand, [[-1.0, 1.0], [-1.0, 1.0], [0, 2 * numpy.pi]], Ksample=Ksample
    )
    n = numpy.arange(0, N)[:, numpy.newaxis, numpy.newaxis]
    l = numpy.arange(0, L)[numpy.newaxis, :, numpy.newaxis]
    m = numpy.arange(0, L)[numpy.newaxis, numpy.newaxis, :]
    K = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)

    Nln = 0.5 * gammaln(l - m + 1) - 0.5 * gammaln(l + m + 1) - (2 * l) * numpy.log(2)
    NN = numpy.e ** (Nln)

    NN[numpy.where(NN == numpy.inf)] = (
        0  ## To account for the fact that m can't be bigger than l
    )

    constants = NN * (2 * l + 1.0) ** 0.5

    lnI = (
        -(8 * l + 6) * numpy.log(2)
        + gammaln(n + 4 * l + 3)
        - gammaln(n + 1)
        - numpy.log(n + 2 * l + 3.0 / 2)
        - 2 * gammaln(2 * l + 3.0 / 2)
    )
    I = -K * (4 * numpy.pi) * numpy.e ** (lnI)
    # Groups as ((2*I**-1) * integrated) * constants; only the FIRST product is
    # commuted so the backend array leads. Multiplication is commutative but not
    # associative in IEEE 754, so the grouping itself is preserved.
    _fac, _con = like(
        integrated,
        2 * (I**-1.0)[numpy.newaxis, :, :, :],
        constants[numpy.newaxis, :, :, :],
    )
    _res = (integrated * _fac) * _con
    Acos, Asin = _res[0], _res[1]

    return Acos, Asin


class _TimeDepDensityNotVectorized(Exception):
    """Raised when a time-dependent density cannot be evaluated as an array over
    its ``t`` argument, so the caller must fall back to a per-timestep loop."""


# Peak-memory budget (bytes) for a single working copy of the coefficient array
# during a time-vectorized build. The vectorized quadrature accumulates an array
# with a leading time axis, so its working set grows linearly with the number of
# time steps; the ``tgrid`` is therefore processed in batches no larger than this
# budget so that building over a very large ``tgrid`` stays memory-bounded (the
# per-time-slice coefficients are independent, so batching is exact). This is a
# module-level constant rather than a public parameter; tests set it small to
# exercise the batched path.
_TIMEDEP_BATCH_BYTES = 32 * 1024**2  # 32 MB


def _timedep_batch_size(Nt, per_time_elems):
    """Number of time steps to process per batch so one working copy of the
    coefficient array stays within ``_TIMEDEP_BATCH_BYTES``.

    Notes
    -----
    - 2026-07-03 - Written - Bovy (UofT)
    """
    per_batch = _TIMEDEP_BATCH_BYTES // (per_time_elems * 8)  # 8 bytes/float64
    return int(min(Nt, max(1, per_batch)))


def _batched_timedep(tgrid, per_time_elems, compute):
    """Run a vectorized-over-time coefficient computation in batches over
    ``tgrid`` to bound peak memory, concatenating the per-batch results.

    ``compute`` takes a (sub-)``tgrid`` and returns ``(Acos, Asin)`` arrays with
    a leading time axis (``Asin`` may be ``None``). ``per_time_elems`` is the
    number of coefficient-array elements per time step, used to size the batches.

    Notes
    -----
    - 2026-07-03 - Written - Bovy (UofT)
    """
    # Construction-time numerical setup: pin to numpy so the time-vectorized
    # quadrature (density evaluations + the namespace-dispatched _C/_xiToR basis)
    # runs on numpy regardless of any forced backend default (byte-identical no-op
    # on the numpy backend).
    # Pure orchestration -- batching and concatenation only. It must NOT pin to
    # numpy: `compute` IS the coefficient quadrature, so pinning here would undo
    # the migration for every time-dependent build.
    Nt = len(tgrid)
    batch = _timedep_batch_size(Nt, per_time_elems)
    if batch >= Nt:  # fits in one go
        return compute(tgrid)
    acos_parts = []
    asin_parts = []
    for start in range(0, Nt, batch):
        Ac, As = compute(tgrid[start : start + batch])
        acos_parts.append(Ac)
        asin_parts.append(As)
    _xp = get_namespace(acos_parts[0])
    Acos = _xp.concat(acos_parts, axis=0)
    Asin = None if asin_parts[0] is None else _xp.concat(asin_parts, axis=0)
    return Acos, Asin


def _timedep_dens_setup(dens, tgrid, numOfParam):
    """Detect the ``use_physical`` keyword for a time-dependent density and
    verify that it is vectorizable over ``t``; return a callable
    ``f(R, z, phi) -> array over tgrid`` using the first ``numOfParam`` spatial
    arguments. Raises ``_TimeDepDensityNotVectorized`` if calling the density
    with ``t=tgrid`` does not return an array matching ``tgrid``.

    Notes
    -----
    - 2026-07-02 - Written - Bovy (UofT)
    """
    t0 = tgrid[0]
    param = [1.0] * numOfParam
    try:
        dens(*param, t=t0, use_physical=False)
    except Exception:
        dens_kw = {}
    else:
        dens_kw = {"use_physical": False}
    try:
        out = numpy.atleast_1d(dens(*param, t=tgrid, **dens_kw))
    except Exception:
        raise _TimeDepDensityNotVectorized()
    if out.shape != numpy.shape(tgrid):
        raise _TimeDepDensityNotVectorized()

    def f(R, z, phi):
        return numpy.asarray(
            dens(*(R, z, phi)[:numOfParam], t=tgrid, **dens_kw), dtype=float
        )

    return f


def _scf_compute_coeffs_spherical_timedep(dens, N, tgrid, a=1.0, radial_order=None):
    """Vectorized-over-time analogue of ``scf_compute_coeffs_spherical``.

    Evaluates the (spherical) density at all times in ``tgrid`` at once, reusing
    a single radial quadrature whose (time-independent) basis is computed only
    once; returns ``Acos`` of shape ``(Nt, N, 1, 1)``. Raises
    ``_TimeDepDensityNotVectorized`` if the density is not vectorizable over t.

    Notes
    -----
    - 2026-07-02 - Written - Bovy (UofT)
    """
    tgrid = numpy.asarray(tgrid, dtype=float)
    numOfParam = 0
    try:
        dens(0, t=tgrid[0])
        numOfParam = 1
    except Exception:
        try:
            dens(0, 0, t=tgrid[0])
            numOfParam = 2
        except Exception:
            numOfParam = 3
    f = _timedep_dens_setup(dens, tgrid, numOfParam)

    def integrand(xi):
        r = _xiToR(xi, a)
        base = a**3.0 * (1 + xi) ** 2.0 * (1 - xi) ** -3.0 * _C(xi, N, 1)[:, 0]
        return f(r, 0.0, 0.0)[:, numpy.newaxis] * base[numpy.newaxis]  # (Nt, N)

    Ksample = [max(N + 1, 20)]
    if radial_order is not None:
        Ksample[0] = radial_order
    integrated = _gaussianQuadrature(integrand, [[-1.0, 1.0]], Ksample=Ksample)
    n = numpy.arange(0, N)
    K = 16 * numpy.pi * (n + 3.0 / 2) / ((n + 2) * (n + 1) * (1 + n * (n + 3.0) / 2.0))
    Acos = numpy.zeros((len(tgrid), N, 1, 1), float)
    Acos[:, :, 0, 0] = 2 * K[numpy.newaxis] * integrated
    return Acos, None


def _scf_compute_coeffs_axi_timedep(
    dens, N, L, tgrid, a=1.0, radial_order=None, costheta_order=None
):
    """Vectorized-over-time analogue of ``scf_compute_coeffs_axi``; returns
    ``Acos`` of shape ``(Nt, N, L, 1)``.

    Notes
    -----
    - 2026-07-02 - Written - Bovy (UofT)
    """
    tgrid = numpy.asarray(tgrid, dtype=float)
    numOfParam = 0
    try:
        dens(0, 0, t=tgrid[0])
        numOfParam = 2
    except Exception:
        numOfParam = 3
    f = _timedep_dens_setup(dens, tgrid, numOfParam)

    def integrand(xi, costheta):
        l = numpy.arange(0, L)[numpy.newaxis, :]
        r = _xiToR(xi, a)
        R = r * numpy.sqrt(1 - costheta**2.0)
        z = r * costheta
        # Router call; byte-identical on numpy to the scipy spelling it replaces.
        PP = assoc_legendre(L, 1, costheta)[..., 0][numpy.newaxis, :]
        dV = (1.0 + xi) ** 2.0 * numpy.power(1.0 - xi, -4.0)
        _CC = _C(xi, N, L)[:, :]
        _pref = like(_CC, a**3 * (1.0 + xi) ** l * (1.0 - xi) ** (l + 1.0))
        phi_nl = _pref * _CC * PP
        base = phi_nl * dV  # (N, L)
        # `f` evaluates the density over tgrid and returns NUMPY, so it would own
        # `f(...) * base` once base is a backend array; anchor it first.
        _ft = like(base, f(R, z, 0.0))
        return _ft[:, numpy.newaxis, numpy.newaxis] * base[numpy.newaxis]

    Ksample = [max(N + 3 * L // 2 + 1, 20), max(L + 1, 20)]
    if radial_order is not None:
        Ksample[0] = radial_order
    if costheta_order is not None:
        Ksample[1] = costheta_order
    integrated = _gaussianQuadrature(integrand, [[-1, 1], [-1, 1]], Ksample=Ksample) * (
        2 * numpy.pi
    )
    n = numpy.arange(0, N)[:, numpy.newaxis]
    l = numpy.arange(0, L)[numpy.newaxis, :]
    K = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    lnI = (
        -(8 * l + 6) * numpy.log(2)
        + gammaln(n + 4 * l + 3)
        - gammaln(n + 1)
        - numpy.log(n + 2 * l + 3.0 / 2)
        - 2 * gammaln(2 * l + 3.0 / 2)
    )
    I = -K * (4 * numpy.pi) * numpy.e ** (lnI)
    constants = -(2.0 ** (-2 * l)) * (2 * l + 1.0) ** 0.5
    # Built functionally; groups as ((2*I**-1) * integrated) * constants with
    # only the first product commuted so the backend array leads.
    _fac, _con = like(integrated, 2 * (I**-1)[numpy.newaxis], constants[numpy.newaxis])
    _xp = get_namespace(integrated)
    Acos = _xp.reshape((integrated * _fac) * _con, (len(tgrid), N, L, 1))
    return Acos, None


def _scf_compute_coeffs_timedep(
    dens, N, L, tgrid, a=1.0, radial_order=None, costheta_order=None, phi_order=None
):
    """Vectorized-over-time analogue of ``scf_compute_coeffs`` (general,
    non-axisymmetric); returns ``(Acos, Asin)`` of shape ``(Nt, N, L, L)`` each.

    Notes
    -----
    - 2026-07-02 - Written - Bovy (UofT)
    """
    tgrid = numpy.asarray(tgrid, dtype=float)
    f = _timedep_dens_setup(dens, tgrid, 3)

    def integrand(xi, costheta, phi):
        l = numpy.arange(0, L)[numpy.newaxis, :, numpy.newaxis]
        m = numpy.arange(0, L)[numpy.newaxis, numpy.newaxis, :]
        r = _xiToR(xi, a)
        R = r * numpy.sqrt(1 - costheta**2.0)
        z = r * costheta
        # Router call; byte-identical on numpy (swapaxes + .T cancel for 2-D).
        PP = assoc_legendre(L, L, costheta)[numpy.newaxis, :, :]
        dV = (1.0 + xi) ** 2.0 * numpy.power(1.0 - xi, -4.0)
        _CC = _C(xi, N, L)[:, :, numpy.newaxis]
        _pref = like(_CC, -(a**3) * (1.0 + xi) ** l * (1.0 - xi) ** (l + 1.0))
        phi_nl = _pref * _CC * PP
        _cs = like(phi_nl, numpy.array([numpy.cos(m * phi), numpy.sin(m * phi)]))
        base = phi_nl[numpy.newaxis, :, :, :] * _cs * dV  # (2, N, L, L)
        # `f` returns NUMPY over tgrid; anchor before it meets the backend base.
        _ft = like(base, f(R, z, phi))
        return _ft[:, None, None, None, None] * base[numpy.newaxis]

    Ksample = [max(N + 3 * L // 2 + 1, 20), max(L + 1, 20), max(L + 1, 20)]
    if radial_order is not None:
        Ksample[0] = radial_order
    if costheta_order is not None:
        Ksample[1] = costheta_order
    if phi_order is not None:
        Ksample[2] = phi_order
    integrated = _gaussianQuadrature(
        integrand, [[-1.0, 1.0], [-1.0, 1.0], [0, 2 * numpy.pi]], Ksample=Ksample
    )  # (Nt, 2, N, L, L)
    n = numpy.arange(0, N)[:, numpy.newaxis, numpy.newaxis]
    l = numpy.arange(0, L)[numpy.newaxis, :, numpy.newaxis]
    m = numpy.arange(0, L)[numpy.newaxis, numpy.newaxis, :]
    K = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    Nln = 0.5 * gammaln(l - m + 1) - 0.5 * gammaln(l + m + 1) - (2 * l) * numpy.log(2)
    NN = numpy.e ** (Nln)
    NN[numpy.where(NN == numpy.inf)] = 0
    constants = NN * (2 * l + 1.0) ** 0.5
    lnI = (
        -(8 * l + 6) * numpy.log(2)
        + gammaln(n + 4 * l + 3)
        - gammaln(n + 1)
        - numpy.log(n + 2 * l + 3.0 / 2)
        - 2 * gammaln(2 * l + 3.0 / 2)
    )
    I = -K * (4 * numpy.pi) * numpy.e ** (lnI)
    _fac, _con = like(
        integrated,
        2 * (I**-1.0)[None, None, :, :, :],
        constants[None, None, :, :, :],
    )
    res = (integrated * _fac) * _con
    return res[:, 0], res[:, 1]


def _cartesian(arraySizes, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arraySizes : list
        list of size of arrays
    out : numpy.ndarray, optional
        Array to place the cartesian product in.

    Returns
    -------
    numpy.ndarray
        2-D array of shape (product(arraySizes), len(arraySizes)) containing cartesian products

    Notes
    -----
    -  2016-06-02 - Obtained from http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """
    arrays = []
    for i in range(len(arraySizes)):
        arrays.append(numpy.arange(0, arraySizes[i]))

    arrays = [numpy.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = numpy.prod([x.size for x in arrays])
    if out is None:
        out = numpy.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:, 0] = numpy.repeat(arrays[0], m)
    if arrays[1:]:
        _cartesian(arraySizes[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


def _gaussianQuadrature(integrand, bounds, Ksample=[20], roundoff=0):
    """
    Numerically take n integrals over a function that returns a float or an array

    Parameters
    ----------
    integrand : function
        The function you're integrating over.
    bounds : list
        The bounds of the integral in the form of [[a_0, b_0], [a_1, b_1], ... , [a_n, b_n]] where a_i is the lower bound and b_i is the upper bound
    Ksample : list, optional
        Number of sample points in the form of [K_0, K_1, ..., K_n] where K_i is the sample point of the ith integral. (default is [20])
    roundoff : float, optional
        if the integral is less than this value, round it to 0. (default is 0)

    Returns
    -------
    numpy.ndarray
        The integral of the function integrand

    Notes
    -----
    - 2016-05-24 - Written - Aladdin Seaifan (UofT)
    """
    ##Maps the sample point and weights
    xp = numpy.zeros((len(bounds), numpy.max(Ksample)), float)
    wp = numpy.zeros((len(bounds), numpy.max(Ksample)), float)
    for i in range(len(bounds)):
        x, w = leggauss(Ksample[i])  ##Calculates the sample points and weights
        a, b = bounds[i]
        xp[i, : Ksample[i]] = 0.5 * (b - a) * x + 0.5 * (b + a)
        wp[i, : Ksample[i]] = 0.5 * (b - a) * w

    ##Determines the shape of the integrand
    s = 0.0
    shape = None
    s_temp = integrand(*numpy.zeros(len(bounds)))
    if type(s_temp).__name__ == numpy.ndarray.__name__:
        shape = s_temp.shape
        s = numpy.zeros(shape, float)

    # gets all combinations of indices from each integrand
    li = _cartesian(Ksample)

    ##Performs the actual integration
    for i in range(li.shape[0]):
        index = (numpy.arange(len(bounds)), li[i])
        # The integrand value LEADS both operations. numpy would otherwise own
        # `weight * value` and `s += value`, and for a grad-tracking torch
        # Tensor numpy resolves that by calling .numpy() on it, which raises.
        # Addition and multiplication are both commutative in IEEE 754, so the
        # numpy path accumulates exactly the same bits in the same order.
        s = integrand(*xp[index]) * numpy.prod(wp[index]) + s

    ##Rounds values that are less than roundoff to zero -- functional (xp.where)
    ##so it traces under jax/torch instead of an in-place item assignment. For
    ##the default roundoff=0 this is an exact no-op (|s| < 0 is never true), so
    ##the numpy result is byte-identical.
    _xp = get_namespace(s)
    return _xp.where(_xp.abs(s) < roundoff, 0.0, s)
