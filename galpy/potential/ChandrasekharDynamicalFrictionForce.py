###############################################################################
#   ChandrasekharDynamicalFrictionForce: Class that implements the
#                                        Chandrasekhar dynamical friction
###############################################################################
import copy
import hashlib

import numpy
from scipy import interpolate, special

from ..backend import (
    coerce_coords,
    concrete_backend_anchor,
    device_of,
    get_namespace,
    host_eval,
    is_backend_array,
    lazy_cond,
    resolve_namespace,
)
from ..backend import special as _backend_special
from ..backend import to_host
from ..backend._namespaces import (
    namespace_from_arrays,
    requires_backend_grad,
    under_trace,
)
from ..backend.interpolate import Spline1D
from ..util import conversion
from .DissipativeForce import DissipativeForce
from .Potential import (
    _check_c,
    _check_potential_list_and_deprecate,
    _evaluateDensities,
    _pot_grad_namespace,
)

_INVSQRTTWO = 1.0 / numpy.sqrt(2.0)
_INVSQRTPI = 1.0 / numpy.sqrt(numpy.pi)


def _traced_host_namespace(dens):
    """The forced backend's namespace when ``dens`` evaluates to traced values
    (a forced backend inside jax.jit / torch.compile), else None: scipy cannot
    tabulate sigma_r there."""
    if resolve_namespace() is numpy:
        return None
    probe = _evaluateDensities(dens, 1.0, 0.0, phi=0.0, t=0.0)
    return get_namespace(probe) if under_trace(probe) else None


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
        dens : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            Potential instance or a combined potential formed using addition (pot1+pot2+…) that represents the density [default: LogarithmicHaloPotential(normalize=1.,q=1.)].
        sigmar : callable, optional
            Function that gives the velocity dispersion as a function of r (has to be in natural units!); if None, computed from the dens potential using the spherical Jeans equation (in galpy.df.jeans) assuming zero anisotropy; if set to a lambda function, *the object cannot be pickled* (so set it to a real function).
        const_lnLambda : bool, optional
            If set to a number, use a constant ln(Lambda) instead with this value.
        minr : float or Quantity, optional
            Minimum r at which to apply dynamical friction: at r < minr, friction is set to zero.
        maxr : float or Quantity, optional
            Maximum r for which sigmar gets interpolated; for best performance set this to the maximum r you will consider.
        nr : int, optional
            Number of radii to use in the interpolation of sigmar (logarithmically spaced between minr and maxr; linearly if minr=0).
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
        dens = _check_potential_list_and_deprecate(dens)
        self._dens_pot = dens
        # The UNDECORATED evaluator, like every other force in the EOM: this is
        # called once per force evaluation, and the decorated public one re-enters
        # unit parsing + @backend_input coercion every time (98,800 boundary
        # crossings for a five-point integration under a forced backend).
        self._dens_host = lambda R, z, phi=0.0, t=0.0: _evaluateDensities(
            self._dens_pot, R, z, phi=phi, t=t
        )
        from ..df import jeans

        default_sigmar = sigmar is None
        if default_sigmar:
            sigmar = lambda x: jeans.sigmar(
                self._dens_pot, x, beta=0.0, use_physical=False
            )
        # sigma_r varies on the scale r, so tabulate it in log r (a linear grid
        # left sigma_r ~50% wrong at r < 0.03 for the default maxr)
        self._sigmar_rs_4interp = (
            numpy.geomspace(self._minr, self._maxr, nr)
            if self._minr > 0.0
            else numpy.linspace(self._minr, self._maxr, nr)
        )
        # For the default (Jeans) sigma_r, every grid radius comes from one
        # cumulative integral rather than one adaptive quadrature each; falls
        # back to the loop when that path does not apply (and always for a
        # user-supplied sigmar, whose cost is the caller's to control).
        # A host potential carrying a gradient: the numpy grid and the per-radius
        # loop both go through scipy, so build the table with the backend Jeans
        # quadrature instead -- differentiable, and as accurate (Spline1D below
        # then takes differentiated values: mode 2).
        grad_xp = _pot_grad_namespace(self._dens_pot) if default_sigmar else None
        if default_sigmar and grad_xp is None:
            grad_xp = _traced_host_namespace(self._dens_pot)
        fast_sigmars = (
            jeans._sigmar_on_grid(self._dens_pot, self._sigmar_rs_4interp, beta=0.0)
            if default_sigmar and grad_xp is None
            else None
        )
        if grad_xp is not None:
            self._sigmars_4interp = jeans.sigmar(
                self._dens_pot,
                grad_xp.asarray(self._sigmar_rs_4interp),
                beta=0.0,
                use_physical=False,
            )
        elif fast_sigmars is not None:
            self._sigmars_4interp = fast_sigmars
        else:
            sigmars = [sigmar(x) for x in self._sigmar_rs_4interp]
            on_backend = [
                s for s in sigmars if under_trace(s) or requires_backend_grad(s)
            ]
            if on_backend:
                # traced or differentiated values (a sigmar built from backend
                # parameters): keep the graph; concrete ones go to numpy as before
                sxp = namespace_from_arrays(on_backend)
                self._sigmars_4interp = sxp.stack(
                    coerce_coords(sxp, *sigmars, device=device_of(*on_backend))
                )
            else:
                self._sigmars_4interp = numpy.array(to_host(sigmars))
                # ... and concrete backend ones still make this a backend force
                _anchor = concrete_backend_anchor(sigmars)
                if _anchor is not None:
                    self._backend_anchor = _anchor
        if not is_backend_array(self._sigmars_4interp) and numpy.any(
            numpy.isnan(self._sigmars_4interp)
        ):
            # Check for case where density is zero, in that case, just
            # paint in the nearest neighbor for the interpolation
            # (doesn't matter in the end, because force = 0 when dens = 0)
            nanrs_indx = numpy.isnan(self._sigmars_4interp)
            if numpy.all(
                numpy.array(
                    to_host(
                        [
                            self._dens_host(r * _INVSQRTTWO, r * _INVSQRTTWO)
                            for r in self._sigmar_rs_4interp[nanrs_indx]
                        ]
                    )
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
        # The natural cubic spline the C implementation (GSL cspline) evaluates;
        # numpy queries call scipy, backend queries evaluate the same
        # polynomial through the namespace (built in-backend for a host with
        # backend parameters, so sigma_r stays differentiable in them).
        self.sigmar = (
            Spline1D(self._sigmar_rs_4interp, self._sigmars_4interp, bc="natural")
            if is_backend_array(self._sigmars_4interp)
            else Spline1D.from_ppoly(
                interpolate.CubicSpline(
                    self._sigmar_rs_4interp, self._sigmars_4interp, bc_type="natural"
                )
            )
        )
        if const_lnLambda:
            self._lnLambda = const_lnLambda
        else:
            self._lnLambda = False
        self._amp = self._amp * (4.0 * numpy.pi)
        self._force_hash = None
        self.hasC = _check_c(self._dens_pot, dens=True)
        # The rectangular force Jacobian (dF/dx, dF/dv) for the 3D variational
        # equations is wired in C (ChandrasekharDynamicalFrictionForce.c); like
        # the force itself it needs the background density (and its gradient,
        # taken by finite differences in C) -> same requirement as hasC.
        self.hasC_dxdv3d = self.hasC
        # No planar version of (Chandrasekhar/FDM) dynamical friction in the
        # C integrator (also covers FDMDynamicalFrictionForce, which inherits
        # its hasC* attributes from here)
        self.hasC_planar = False
        self._backend_compatible = True
        self._reset_param_namespace()
        return None

    def GMs(self, gms):
        gms = conversion.parse_mass(gms, ro=self._ro, vo=self._vo)
        self._amp = self._amp * (gms / self._ms)
        self._ms = gms
        # Reset the hash
        self._force_hash = None
        self._reset_param_namespace()
        return None

    GMs = property(None, GMs)

    def rhm(self, new_rhm):
        self._rhm = conversion.parse_length(new_rhm, ro=self._ro)
        # Reset the hash
        self._force_hash = None
        self._reset_param_namespace()
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
        xp = resolve_namespace(r, v)
        if xp is not numpy:
            return self._lnLambda_backend(*coerce_coords(xp, r, v), xp)
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

    def _lnLambda_backend(self, r, v, xp):
        # jit/grad-safe Coulomb logarithm for backend (jax/torch) inputs; value
        # matches lnLambda(). The rhm/GM-over-v^2 selection is an xp.where; the
        # rhm==0 (black-hole default) case is handled in Python (static attr) so
        # the r/gamma/rhm dead branch (division by zero) never poisons autodiff.
        if self._lnLambda:
            return self._lnLambda
        GMvs = self._ms / v**2.0
        if under_trace(self._rhm):
            # rhm is a fit parameter here, so rhm == 0.0 has no concrete value.
            # The rhm==0 arm is the GMvs>=rhm branch of the where below with the
            # 1/rhm removed, so take the where and guard that denominator -- an
            # unguarded 1/0 would NaN-poison d/d(rhm) everywhere.
            _rhm = xp.where(self._rhm == 0.0, 1.0, self._rhm)
            Lambda = xp.where(
                GMvs < self._rhm,
                r / self._gamma / _rhm,
                r / self._gamma / GMvs,
            )
        elif self._rhm == 0.0:
            Lambda = r / self._gamma / GMvs
        else:
            Lambda = xp.where(
                GMvs < self._rhm,
                r / self._gamma / self._rhm,
                r / self._gamma / GMvs,
            )
        return 0.5 * xp.log(1.0 + Lambda**2.0)

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
                -self._dens_host(R, z, phi=phi, t=t) / vs**3.0 * Xfactor * lnLambda
            )

    def _sigmar_backend(self, r, xp):
        """sigma_r(r) on the backend: the table spline inside maxr, the given
        sigmar (sigmar_orig) beyond, as the numpy path does."""
        sr = self.sigmar(r)
        outside = r > self._maxr

        def beyond():
            # dead-branch-safe: sigmar_orig only sees radii beyond maxr
            r_out = xp.where(outside, r, self._maxr * xp.ones_like(r))
            try:
                s_out = self.sigmar_orig(r_out)
            except Exception:
                s_out = None
            if s_out is None or isinstance(s_out, numpy.ndarray):
                # a numpy-only user sigmar: its values (no gradient)
                s_out = host_eval(self.sigmar_orig, r_out)
            return xp.where(outside, s_out * xp.ones_like(sr), sr)

        # sigmar_orig (for the default, a Jeans integral) only when needed
        return lazy_cond(xp, xp.any(outside), beyond, lambda: sr)

    def _calc_force_backend(self, R, phi, z, v, t, xp):
        # jit/grad-safe common friction factor for backend inputs; no hashing/
        # caching (traced arrays are unhashable). Returns the scalar that the
        # cylindrical force components are built from.
        # v is a caller-supplied triple, not coerced by @backend_input like R/z,
        # and R/z are plain floats when only a parameter is on the backend.
        R, z, phi, t, v = self._coerce_force_args(R, z, phi, t, v, xp)
        r = xp.sqrt(R**2.0 + z**2.0)
        vs = xp.sqrt(v[0] ** 2.0 + v[1] ** 2.0 + v[2] ** 2.0)
        sr = self._sigmar_backend(r, xp)
        X = vs * _INVSQRTTWO / sr
        Xfactor = _backend_special.erf(X) - 2.0 * X * _INVSQRTPI * xp.exp(-(X**2.0))
        lnLambda = self._lnLambda_backend(r, vs, xp)
        force = -self._dens_host(R, z, phi=phi, t=t) / vs**3.0 * Xfactor * lnLambda
        return xp.where(r < self._minr, xp.zeros_like(force), force)

    def _cached_force_factor(self, R, phi, z, v, t):
        """The shared friction factor, recomputed only when (R, phi, z, v, t) moves.

        The three force components are queried with identical arguments inside one
        equation-of-motion evaluation, so the factor -- and the host-density
        evaluation behind it -- is computed once per step rather than three times.
        The hash was previously computed and compared but never STORED, so
        ``self._force_hash`` stayed ``None`` and the cache never hit.

        Backend inputs never reach here: each component takes the
        ``_calc_force_backend`` branch first, which is uncached by design (traced
        arrays are unhashable).
        """
        new_hash = hashlib.md5(
            numpy.array([R, phi, z, v[0], v[1], v[2], t])
        ).hexdigest()
        if new_hash != self._force_hash:
            self._calc_force(R, phi, z, v, t)
            self._force_hash = new_hash
        return self._cached_force

    def _backend_force_factor(self, R, phi, z, v, t, xp):
        """``_calc_force_backend`` with the eager twin of ``_cached_force_factor``:
        the three components of one EOM evaluation pass the SAME arrays, so the
        factor is computed once, not three times. Keyed on object identity (and
        torch's in-place version counter), so a different or modified input
        always recomputes; skipped under a trace (no stateful caching there)."""
        args = (R, phi, z, t, v[0], v[1], v[2])
        if under_trace(*args):
            return self._calc_force_backend(R, phi, z, v, t, xp)
        key = tuple((a, getattr(a, "_version", None)) for a in args)
        hit = getattr(self, "_backend_force_cache", None)
        if hit is not None and all(
            k[0] is a and k[1] == ka for (k, (a, ka)) in zip(hit[0], key)
        ):
            return hit[1]
        out = self._calc_force_backend(R, phi, z, v, t, xp)
        self._backend_force_cache = (key, out)
        return out

    def _reset_param_namespace(self):
        """Cache the namespace of a differentiated parameter (amp, GMs, gamma,
        rhm, the host potential, a user sigmar), else None; called whenever a
        parameter is set. The eager force cache holds coordinates, not
        parameters, so it goes first."""
        self.__dict__.pop("_backend_force_cache", None)
        self.__dict__.pop("_param_xp", None)
        # gradient-carrying only: a merely-forced backend leaves backend-array
        # host parameters behind, and the Python integrators' numpy steps must
        # stay on numpy
        self._param_xp = _pot_grad_namespace(self)

    def _param_namespace(self):
        try:
            return self._param_xp
        except AttributeError:  # unpickled
            self._reset_param_namespace()
            return self._param_xp

    def _force_namespace(self, R, z, phi, t, v):
        # Dispatch on the DATA, not get_namespace: this DissipativeForce is queried
        # via a path that skips the input-coercion gate, so under a forced backend
        # get_namespace would return that backend for bare python/numpy inputs and
        # send them into the backend arithmetic (torch.sqrt(float) crashes). Backend
        # coordinates or a differentiated parameter take the backend path; else
        # (None) the numpy/cache path.
        if is_backend_array(R) or is_backend_array(z) or any(map(is_backend_array, v)):
            return get_namespace(R, z, phi, t, v[0], v[1], v[2])
        return self._param_namespace()

    def _coerce_force_args(self, R, z, phi, t, v, xp):
        """(R, z, phi, t, v) on ``xp``, anchored on the coordinates' device or,
        for plain-float coordinates, on the parameters'."""
        dev = device_of(R, z, *v) or device_of(
            self._amp, self._ms, self._gamma, self._rhm, self._sigmars_4interp
        )
        R, z, phi, t = coerce_coords(xp, R, z, phi, t, device=dev)
        return R, z, phi, t, coerce_coords(xp, *v, device=dev)

    def _Rforce(self, R, z, phi=0.0, t=0.0, v=None):
        xp = self._force_namespace(R, z, phi, t, v)
        if xp is not None:
            return self._backend_force_factor(R, phi, z, v, t, xp) * v[0]
        return self._cached_force_factor(R, phi, z, v, t) * v[0]

    def _phitorque(self, R, z, phi=0.0, t=0.0, v=None):
        xp = self._force_namespace(R, z, phi, t, v)
        if xp is not None:
            return self._backend_force_factor(R, phi, z, v, t, xp) * v[1] * R
        return self._cached_force_factor(R, phi, z, v, t) * v[1] * R

    def _zforce(self, R, z, phi=0.0, t=0.0, v=None):
        xp = self._force_namespace(R, z, phi, t, v)
        if xp is not None:
            return self._backend_force_factor(R, phi, z, v, t, xp) * v[2]
        return self._cached_force_factor(R, phi, z, v, t) * v[2]

    # Pickling functions
    def __getstate__(self):
        pdict = copy.copy(self.__dict__)
        # rm lambda function
        del pdict["_dens_host"]
        pdict.pop("_backend_force_cache", None)  # holds arrays of the last step
        pdict.pop("_param_xp", None)  # a module
        if self._sigmar_kwarg is None:
            # because an object set up with sigmar = user-provided function
            # cannot typically be picked, disallow this explicitly
            # (so if it can, everything should be fine; if not, pickling error)
            del pdict["sigmar_orig"]
        return pdict

    def __setstate__(self, pdict):
        self.__dict__ = pdict
        # Re-setup _dens_host
        # The UNDECORATED evaluator, like every other force in the EOM: this is
        # called once per force evaluation, and the decorated public one re-enters
        # unit parsing + @backend_input coercion every time (98,800 boundary
        # crossings for a five-point integration under a forced backend).
        self._dens_host = lambda R, z, phi=0.0, t=0.0: _evaluateDensities(
            self._dens_pot, R, z, phi=phi, t=t
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
