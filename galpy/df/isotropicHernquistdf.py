# Class that implements isotropic spherical Hernquist DF
# computed using the Eddington formula
import numpy

from ..backend import resolve_namespace
from ..potential import HernquistPotential, evaluatePotentials
from ..util import conversion
from .sphericaldf import isotropicsphericaldf


class isotropicHernquistdf(isotropicsphericaldf):
    """Class that implements isotropic spherical Hernquist DF computed using the Eddington formula"""

    def __init__(self, pot=None, ro=None, vo=None):
        """
        Initialize an isotropic Hernquist distribution function.

        Parameters
        ----------
        pot : HernquistPotential instance
            Hernquist potential instance.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-08-09 - Written - Lane (UofT)
        """
        assert isinstance(pot, HernquistPotential), (
            "pot= must be potential.HernquistPotential"
        )
        isotropicsphericaldf.__init__(self, pot=pot, ro=ro, vo=vo)
        self._psi0 = -evaluatePotentials(self._pot, 0, 0, use_physical=False)
        self._GMa = self._psi0 * self._pot.a**2.0
        # first factor = mass to make the DF that of mass density
        self._fEnorm = (
            self._psi0
            * self._pot.a
            / numpy.sqrt(2.0)
            / (2 * numpy.pi) ** 3
            / self._GMa**1.5
        )

    def fE(self, E):
        """
        Calculate the energy portion of an isotropic Hernquist distribution function

        Parameters
        ----------
        E : float or Quantity
            The energy.

        Returns
        -------
        numpy.ndarray
            The value of the energy portion of the DF.

        Notes
        -----
        - 2020-08-09 - Written - James Lane (UofT)
        """
        Ei = conversion.parse_energy(E, vo=self._vo)
        # resolve on the stored _psi0 too: a backend-built potential makes the
        # DF constants backend so parameter gradients keep flowing
        xp = resolve_namespace(Ei, self._psi0)
        if xp is numpy:
            Etilde = -numpy.atleast_1d(Ei / self._psi0)
            # Handle E out of bounds
            Etilde_out = numpy.where(numpy.logical_or(Etilde < 0, Etilde > 1))[0]
            if len(Etilde_out) > 0:
                # Set to dummy and 0 later, prevents functions throwing errors?
                Etilde[Etilde_out] = 0.5
            sqrtEtilde = numpy.sqrt(Etilde)
            fE = (
                self._fEnorm
                * sqrtEtilde
                / (1 - Etilde) ** 2.0
                * (
                    (1.0 - 2.0 * Etilde) * (8.0 * Etilde**2.0 - 8.0 * Etilde - 3.0)
                    + (
                        (3.0 * numpy.arcsin(sqrtEtilde))
                        / numpy.sqrt(Etilde * (1.0 - Etilde))
                    )
                )
            )
            # Fix out of bounds values
            if len(Etilde_out) > 0:
                fE[Etilde_out] = 0
            return fE.reshape(E.shape)
        # jax/torch: functional out-of-bounds handling (same dummy-then-zero,
        # with the dummy keeping the dead branch NaN-free under autodiff); the
        # Etilde == 0 edge is 0/0 in the arcsin term but fE -> 0 (~ Etilde^{5/2})
        Eb = xp.asarray(Ei) * 1.0
        Etilde = -xp.atleast_1d(Eb) / self._psi0
        dead = (Etilde < 0) | (Etilde > 1) | (Etilde == 0)
        Etilde = xp.where(dead, 0.5, Etilde)
        sqrtEtilde = xp.sqrt(Etilde)
        fE = (
            self._fEnorm
            * sqrtEtilde
            / (1 - Etilde) ** 2.0
            * (
                (1.0 - 2.0 * Etilde) * (8.0 * Etilde**2.0 - 8.0 * Etilde - 3.0)
                + ((3.0 * xp.arcsin(sqrtEtilde)) / xp.sqrt(Etilde * (1.0 - Etilde)))
            )
        )
        fE = xp.where(dead, xp.zeros_like(fE), fE)
        return fE.reshape(Eb.shape)

    def _dMdE(self, E):
        # E already in internal units here
        xp = resolve_namespace(E, self._psi0)
        if xp is numpy:
            fE = self.fE(E)
            A = -self._psi0 / E[fE > 0.0]
            out = numpy.zeros_like(E)
            out[fE > 0.0] = (
                (4.0 * numpy.pi) ** 2.0
                * fE[fE > 0.0]
                * self._pot.a**3.0
                * numpy.sqrt(-2.0 * E[fE > 0.0])
                * (
                    numpy.sqrt(A - 1.0) * (0.125 * A**2.0 - 5.0 / 12.0 * A - 1.0 / 3.0)
                    + 0.125 * A * (A**2.0 - 4.0 * A + 8.0) * numpy.arccos(A**-0.5)
                )
            )
            return out
        # jax/torch: functional masking; out-of-bounds E gets the safe dummy
        # A = 2 (finite sqrt/arccos on the dead branch under autodiff)
        fE = self.fE(E)
        pos = fE > 0.0
        Esafe = xp.where(pos, xp.asarray(E) * 1.0, -self._psi0 / 2.0)
        A = -self._psi0 / Esafe
        out = (
            (4.0 * numpy.pi) ** 2.0
            * fE
            * self._pot.a**3.0
            * xp.sqrt(-2.0 * Esafe)
            * (
                xp.sqrt(A - 1.0) * (0.125 * A**2.0 - 5.0 / 12.0 * A - 1.0 / 3.0)
                + 0.125 * A * (A**2.0 - 4.0 * A + 8.0) * xp.arccos(A**-0.5)
            )
        )
        return xp.where(pos, out, xp.zeros_like(out))

    def _icmf(self, ms):
        """Analytic expression for the normalized inverse cumulative mass
        function. The argument ms is normalized mass fraction [0,1]"""
        xp = resolve_namespace(ms)
        if xp is numpy:
            return self._pot.a * numpy.sqrt(ms) / (1 - numpy.sqrt(ms))
        sq = xp.sqrt(xp.asarray(ms) * 1.0)  # coerce: torch.sqrt rejects numpy
        return self._pot.a * sq / (1 - sq)
