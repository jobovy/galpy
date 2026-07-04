# Class that implements the anisotropic spherical Hernquist DF with constant
# beta parameter
import numpy
import scipy.integrate
import scipy.special

from ..backend import resolve_namespace
from ..backend import special as bspecial
from ..potential import HernquistPotential, evaluatePotentials
from ..util import conversion
from .constantbetadf import _constantbetadf


class constantbetaHernquistdf(_constantbetadf):
    """Class that implements the anisotropic spherical Hernquist DF with constant beta parameter"""

    def __init__(self, pot=None, beta=0, ro=None, vo=None):
        """
        Initialize a Hernquist DF with constant anisotropy.

        Parameters
        ----------
        pot : HernquistPotential
            Hernquist potential which determines the DF.
        beta : float
            Anisotropy parameter.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-07-22 - Written - Lane (UofT)
        """
        assert isinstance(pot, HernquistPotential), (
            "pot= must be potential.HernquistPotential"
        )
        _constantbetadf.__init__(self, pot=pot, beta=beta, ro=ro, vo=vo)
        self._psi0 = -evaluatePotentials(self._pot, 0, 0, use_physical=False)
        self._potInf = 0.0
        self._GMa = self._psi0 * self._pot.a**2.0
        # Final factor is mass to make the DF that of the mass density
        self._fEnorm = (
            (2.0**self._beta / (2.0 * numpy.pi) ** 2.5)
            * scipy.special.gamma(5.0 - 2.0 * self._beta)
            / scipy.special.gamma(1.0 - self._beta)
            / scipy.special.gamma(3.5 - self._beta)
            / self._GMa ** (1.5 - self._beta)
            * self._psi0
            * self._pot.a
        )

    def fE(self, E):
        """
        Calculate the energy portion of a Hernquist distribution function

        Parameters
        ----------
        E : float, numpy.ndarray, or Quantity
            The energy.

        Returns
        -------
        float or numpy.ndarray
            The value of the energy portion of the DF

        Notes
        -----
        - 2020-07-22 - Written

        """
        Ei = conversion.parse_energy(E, vo=self._vo)
        # resolve on _psi0 too so backend-built potential params keep gradients
        xp = resolve_namespace(Ei, self._psi0)
        if xp is numpy:
            Etilde = -numpy.atleast_1d(Ei / self._psi0)
            # Handle potential E outside of bounds
            Etilde_out = numpy.where(numpy.logical_or(Etilde < 0, Etilde > 1))[0]
            if len(Etilde_out) > 0:
                # Dummy variable now and 0 later, prevents numerical issues?
                Etilde[Etilde_out] = 0.5
            # First check algebraic solutions, all adjusted such that DF = mass den
            if self._beta == 0.0:  # isotropic case
                sqrtEtilde = numpy.sqrt(Etilde)
                fE = (
                    self._psi0
                    * self._pot.a
                    / numpy.sqrt(2.0)
                    / (2 * numpy.pi) ** 3
                    / self._GMa**1.5
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
            elif self._beta == 0.5:
                fE = (3.0 * Etilde**2.0) / (4.0 * numpy.pi**3.0 * self._pot.a)
            elif self._beta == -0.5:
                fE = (
                    (20.0 * Etilde**3.0 - 20.0 * Etilde**4.0 + 6.0 * Etilde**5.0)
                    / (1.0 - Etilde) ** 4
                ) / (4.0 * numpy.pi**3.0 * self._GMa * self._pot.a)
            else:
                fE = (
                    self._fEnorm
                    * numpy.power(Etilde, 2.5 - self._beta)
                    * scipy.special.hyp2f1(
                        5.0 - 2.0 * self._beta,
                        1.0 - 2.0 * self._beta,
                        3.5 - self._beta,
                        Etilde,
                    )
                )
            if len(Etilde_out) > 0:
                fE[Etilde_out] = 0.0
            return fE.reshape(E.shape)
        # jax/torch: functional out-of-bounds handling (dummy-then-zero, keeping
        # the dead branch NaN-free under autodiff); beta==0 is 0/0 at Etilde==0
        Eb = xp.asarray(Ei) * 1.0
        Etilde = -xp.atleast_1d(Eb) / self._psi0
        dead = (Etilde < 0) | (Etilde > 1)
        if self._beta == 0.0:
            dead = dead | (Etilde == 0)
        Etilde = xp.where(dead, 0.5, Etilde)
        if self._beta == 0.0:  # isotropic case
            sqrtEtilde = xp.sqrt(Etilde)
            fE = (
                self._psi0
                * self._pot.a
                / numpy.sqrt(2.0)
                / (2 * numpy.pi) ** 3
                / self._GMa**1.5
                * sqrtEtilde
                / (1 - Etilde) ** 2.0
                * (
                    (1.0 - 2.0 * Etilde) * (8.0 * Etilde**2.0 - 8.0 * Etilde - 3.0)
                    + ((3.0 * xp.arcsin(sqrtEtilde)) / xp.sqrt(Etilde * (1.0 - Etilde)))
                )
            )
        elif self._beta == 0.5:
            fE = (3.0 * Etilde**2.0) / (4.0 * numpy.pi**3.0 * self._pot.a)
        elif self._beta == -0.5:
            fE = (
                (20.0 * Etilde**3.0 - 20.0 * Etilde**4.0 + 6.0 * Etilde**5.0)
                / (1.0 - Etilde) ** 4
            ) / (4.0 * numpy.pi**3.0 * self._GMa * self._pot.a)
        else:
            # 2F1(5-2b,1-2b;3.5-b;Etilde) with Etilde in [0,1] (positive z); the
            # backend hyp2f1 fallback is built for z<=0, so use the Pfaff transform
            # 2F1(a,b;c;z)=(1-z)^{-b} 2F1(c-a,b;c;z/(z-1)) to map z into (-inf,0]
            a = 5.0 - 2.0 * self._beta
            b = 1.0 - 2.0 * self._beta  # > 0 requires beta < 0.5 for the fallback
            c = 3.5 - self._beta
            fE = (
                self._fEnorm
                * Etilde ** (2.5 - self._beta)
                * (1.0 - Etilde) ** (-b)
                * bspecial.hyp2f1(c - a, b, c, Etilde / (Etilde - 1.0))
            )
        fE = xp.where(dead, xp.zeros_like(fE), fE)
        return fE.reshape(Eb.shape)

    def _icmf(self, ms):
        """Analytic expression for the normalized inverse cumulative mass
        function. The argument ms is normalized mass fraction [0,1]"""
        xp = resolve_namespace(ms)
        if xp is numpy:
            return self._pot.a * numpy.sqrt(ms) / (1 - numpy.sqrt(ms))
        sq = xp.sqrt(xp.asarray(ms) * 1.0)  # coerce: torch.sqrt rejects numpy
        return self._pot.a * sq / (1 - sq)
