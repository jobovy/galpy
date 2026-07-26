###############################################################################
#   RazorThinExponentialDiskPotential.py: class that implements the razor thin
#                                         exponential disk potential
#
#                                      rho(R,z) = rho_0 e^-R/h_R delta(z)
###############################################################################
import math

import numpy

from ..backend import coerce_coords, get_namespace
from ..backend import special as bspecial
from ..util import conversion
from .Potential import Potential


class RazorThinExponentialDiskPotential(Potential):
    """Class that implements the razor-thin exponential disk potential

    .. math::

        \\rho(R,z) = \\mathrm{amp}\\,\\exp\\left(-R/h_R\\right)\\,\\delta(z)

    """

    def __init__(
        self,
        amp=1.0,
        hr=1.0 / 3.0,
        normalize=False,
        ro=None,
        vo=None,
        new=True,
        glorder=100,
    ):
        """
        Class that implements a razor-thin exponential disk potential.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential (default: 1); can be a Quantity with units of surface-mass or Gxsurface-mass.
        hr : float or Quantity, optional
            Disk scale-length.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        new : bool, optional
            If True, use a new implementation of the potential that is more accurate for small scale lengths (default: True).
        glorder : int, optional
            Gaussian quadrature order to use for numerical integration (default: 100).

        Notes
        -----
        - 2012-12-27 - Written - Bovy (IAS)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units="surfacedensity")
        hr = conversion.parse_length(hr, ro=self._ro)
        self._new = new
        self._glorder = glorder
        self._hr = hr
        self._scale = self._hr
        self._alpha = 1.0 / self._hr
        self._glx, self._glw = numpy.polynomial.legendre.leggauss(self._glorder)
        self._backend_compatible = True
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)

    def _inner_nodes(self, R):
        """Substituted Gauss-Legendre nodes/weights for the inner panel [0, R].

        As ``|z| -> 0`` the force integrands carry a factor
        ``1/sqrt(R^2+z^2-k^2+sqrtp*sqrtm)`` with a square-root singularity
        exactly at the panel edge ``k=R``, against which fixed-order
        Gauss-Legendre converges only algebraically -- n=3000 still leaves 1e-4.
        Substituting ``k = R(1-v^2)`` puts the Jacobian's zero (``2Rv``)
        precisely where that factor blows up, so the product is smooth and the
        same 100 nodes reach ~5e-5 instead of ~5e-3: better than n=3000, at no
        extra integrand evaluations. Weights follow this module's convention of
        carrying the full panel width rather than half of it.
        """
        # asarray first: R may be a backend array, and a torch tensor times a
        # numpy array trips numpy's __array_wrap__ (no-op, byte-identical, on numpy)
        xp = get_namespace(R)
        glx = xp.asarray(self._glx)
        glw = xp.asarray(self._glw)
        v = 0.5 * (glx + 1.0)
        return R * (1.0 - v**2.0), 2.0 * R * v * glw

    def _outer_nodes(self, R, kmax):
        """The same, for the outer panel [R, kmax], via ``k = R + u^2``."""
        xp = get_namespace(R)
        glx = xp.asarray(self._glx)
        glw = xp.asarray(self._glw)
        umax = xp.sqrt(kmax - R)
        u = umax * 0.5 * (glx + 1.0)
        return R + u**2.0, 2.0 * umax * u * glw

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        if self._new:
            # In-plane closed form vs. the Gauss-Legendre integral: xp.where,
            # not a python `if`, so this traces. Both branches run eagerly, so
            # each gets a dead-side guard (Bessel at R=0, integrand at z=0).
            inplane = xp.abs(z) < 10.0**-6.0
            R_safe = xp.where(inplane, R, xp.ones_like(R * 1.0))
            z_safe = xp.where(inplane, xp.ones_like(z * 1.0), z)
            y = 0.5 * self._alpha * R_safe
            inplane_val = (
                -math.pi
                * R_safe
                * (bspecial.i0(y) * bspecial.k1(y) - bspecial.i1(y) * bspecial.k0(y))
            )
            kalphamax = 10.0
            # ks/weights are built from the float64 Gauss-Legendre nodes; move
            # them onto the active backend/device anchored on the inputs so that
            # ks + R etc. are same-namespace (a numpy ndarray + a torch tensor
            # raises). xp.asarray on the numpy path is a no-op (byte-identical).
            ks = xp.asarray(kalphamax * 0.5 * (self._glx + 1.0))
            weights = xp.asarray(kalphamax * self._glw)
            # The nodes broadcast on a NEW TRAILING axis and the quadrature
            # reduces only that axis, so the coordinate shape is preserved.
            # Reducing every axis instead collapses the DATA axis too and
            # returns one Phi for every point -- silently, and galpy's Phi call
            # IS vectorised (Orbit.E(ts) evaluates all times at once), so that
            # is a wrong answer rather than an error.
            Rb = xp.asarray(R)[..., None]
            zb = xp.asarray(z_safe)[..., None]
            sqrtp = xp.sqrt(zb**2.0 + (ks + Rb) ** 2.0)
            sqrtm = xp.sqrt(zb**2.0 + (ks - Rb) ** 2.0)
            evalInt = (
                xp.arcsin(2.0 * ks / (sqrtp + sqrtm))
                * ks
                * bspecial.k0(self._alpha * ks)
            )
            return xp.where(
                inplane,
                inplane_val,
                -2.0 * self._alpha * xp.sum(weights * evalInt, axis=-1),
            )
        raise NotImplementedError(
            "Not new=True not implemented for RazorThinExponentialDiskPotential"
        )

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        # move the numpy GL nodes/weights onto the backend first, so products
        # like R * glx are same-namespace (a torch tensor * a numpy array trips
        # numpy's __array_wrap__); xp.asarray is a no-op on numpy (byte-identical)
        glx = xp.asarray(self._glx)
        glw = xp.asarray(self._glw)
        if self._new:
            # if R > 6.: return self._kp(R,z)
            # xp.where in place of the data-dependent `if`s: in-plane closed
            # form (R_safe: Bessel diverges at R=0) vs the [0,R]+[R,10]
            # quadrature (z_safe: its integrand is 0/0 at z=0 for k>R).
            inplane = xp.abs(z) < 10.0**-6.0
            R_safe = xp.where(inplane, R, xp.ones_like(R * 1.0))
            z_safe = xp.where(inplane, xp.ones_like(z * 1.0), z)
            y = 0.5 * self._alpha * R_safe
            inplane_val = (
                -2.0
                * math.pi
                * y
                * (bspecial.i0(y) * bspecial.k0(y) - bspecial.i1(y) * bspecial.k1(y))
            )
            # Nodes live on a NEW TRAILING axis and the quadrature reduces only
            # that axis, so the coordinate shape is preserved (see _evaluate).
            Rb = xp.asarray(R)[..., None]
            zb = xp.asarray(z_safe)[..., None]
            # main's substituted nodes (k = R(1-v^2)): the Jacobian's zero sits on
            # the k=R square-root singularity, ~5e-5 instead of ~5e-3. Rb carries the
            # trailing node axis, so the helpers broadcast to (..., n) unchanged.
            ks1, weights1 = self._inner_nodes(Rb)
            sqrtp = xp.sqrt(zb**2.0 + (ks1 + Rb) ** 2.0)
            sqrtm = xp.sqrt(zb**2.0 + (ks1 - Rb) ** 2.0)
            evalInt1 = (
                ks1**2.0
                * bspecial.k0(ks1 * self._alpha)
                * ((ks1 + Rb) / sqrtp - (ks1 - Rb) / sqrtm)
                / xp.sqrt(Rb**2.0 + zb**2.0 - ks1**2.0 + sqrtp * sqrtm)
                / (sqrtp + sqrtm)
            )
            # [R,10] panel: max(R,10) gives it zero width -- so exactly zero
            # weights -- for R >= 10, replacing the old `if R < 10.`
            kalphamax2 = xp.maximum(Rb, 10.0 * xp.ones_like(Rb * 1.0))
            # k = R + u^2, umax = sqrt(kmax-R): at R >= 10 kmax == Rb so umax == 0 and
            # the weights are EXACTLY zero -- the same empty-panel trick as the linear
            # map it replaces (verified 0.0, not merely small).
            ks2, weights2 = self._outer_nodes(Rb, kalphamax2)
            sqrtp = xp.sqrt(zb**2.0 + (ks2 + Rb) ** 2.0)
            sqrtm = xp.sqrt(zb**2.0 + (ks2 - Rb) ** 2.0)
            evalInt2 = (
                ks2**2.0
                * bspecial.k0(ks2 * self._alpha)
                * ((ks2 + Rb) / sqrtp - (ks2 - Rb) / sqrtm)
                / xp.sqrt(Rb**2.0 + zb**2.0 - ks2**2.0 + sqrtp * sqrtm)
                / (sqrtp + sqrtm)
            )
            return xp.where(
                inplane,
                inplane_val,
                -2.0
                * math.sqrt(2.0)
                * self._alpha
                * xp.sum(weights1 * evalInt1 + weights2 * evalInt2, axis=-1),
            )
        raise NotImplementedError(
            "Not new=True not implemented for RazorThinExponentialDiskPotential"
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        glx = xp.asarray(self._glx)
        glw = xp.asarray(self._glw)
        if self._new:
            # if R > 6.: return self._kp(R,z)
            # zforce = 0 in the plane, where the integrand has 1/|z| poles ->
            # z_safe guards the eagerly-evaluated dead branch; max(R,10) empties
            # the [R,10] panel for R >= 10 (the old `if R < 10.`).
            inplane = xp.abs(z) < 10.0**-6.0
            z_safe = xp.where(inplane, xp.ones_like(z * 1.0), z)
            # Nodes on a NEW TRAILING axis, reduce only that axis (see _evaluate).
            Rb = xp.asarray(R)[..., None]
            zb = xp.asarray(z_safe)[..., None]
            # main's substituted nodes (k = R(1-v^2)): the Jacobian's zero sits on
            # the k=R square-root singularity, ~5e-5 instead of ~5e-3. Rb carries the
            # trailing node axis, so the helpers broadcast to (..., n) unchanged.
            ks1, weights1 = self._inner_nodes(Rb)
            sqrtp = xp.sqrt(zb**2.0 + (ks1 + Rb) ** 2.0)
            sqrtm = xp.sqrt(zb**2.0 + (ks1 - Rb) ** 2.0)
            evalInt1 = (
                ks1**2.0
                * bspecial.k0(ks1 * self._alpha)
                * (1.0 / sqrtp + 1.0 / sqrtm)
                / xp.sqrt(Rb**2.0 + zb**2.0 - ks1**2.0 + sqrtp * sqrtm)
                / (sqrtp + sqrtm)
            )
            kalphamax2 = xp.maximum(Rb, 10.0 * xp.ones_like(Rb * 1.0))
            # k = R + u^2, umax = sqrt(kmax-R): at R >= 10 kmax == Rb so umax == 0 and
            # the weights are EXACTLY zero -- the same empty-panel trick as the linear
            # map it replaces (verified 0.0, not merely small).
            ks2, weights2 = self._outer_nodes(Rb, kalphamax2)
            sqrtp = xp.sqrt(zb**2.0 + (ks2 + Rb) ** 2.0)
            sqrtm = xp.sqrt(zb**2.0 + (ks2 - Rb) ** 2.0)
            evalInt2 = (
                ks2**2.0
                * bspecial.k0(ks2 * self._alpha)
                * (1.0 / sqrtp + 1.0 / sqrtm)
                / xp.sqrt(Rb**2.0 + zb**2.0 - ks2**2.0 + sqrtp * sqrtm)
                / (sqrtp + sqrtm)
            )
            return xp.where(
                inplane,
                xp.zeros_like(z * 1.0),
                -z
                * 2.0
                * math.sqrt(2.0)
                * self._alpha
                * xp.sum(weights1 * evalInt1 + weights2 * evalInt2, axis=-1),
            )
        raise NotImplementedError(
            "Not new=True not implemented for RazorThinExponentialDiskPotential"
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        if self._new:
            if xp.abs(z) < 10.0**-6.0:
                y = 0.5 * self._alpha * R
                return math.pi * self._alpha * (
                    bspecial.i0(y) * bspecial.k0(y) - bspecial.i1(y) * bspecial.k1(y)
                ) + math.pi / 4.0 * self._alpha**2.0 * R * (
                    bspecial.i1(y) * (3.0 * bspecial.k0(y) + bspecial.kn(2, y))
                    - bspecial.k1(y) * (3.0 * bspecial.i0(y) + bspecial.iv(2, y))
                )
            raise AttributeError(
                "'R2deriv' for RazorThinExponentialDisk not implemented for z =/= 0"
            )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):  # pragma: no cover
        return math.inf

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z)
        R, z = coerce_coords(xp, R, z)
        return xp.exp(-self._alpha * R)

    def _mass(self, R, z=None, t=0.0):
        xp = get_namespace(R)
        (R,) = coerce_coords(xp, R)
        return (
            2.0
            * math.pi
            * (1.0 - xp.exp(-self._alpha * R) * (1.0 + self._alpha * R))
            / self._alpha**2.0
        )
