###############################################################################
#   SymbolicSphericalPotential.py: base class for symbolic potentials
#   corresponding to spherical density profiles
###############################################################################
import numpy

from ..util._optional_deps import _NUMBA_LOADED, _SYMPY_LOADED
from .SphericalPotential import SphericalPotential

if _SYMPY_LOADED:
    import sympy
    from sympy.utilities.autowrap import ufuncify

if _NUMBA_LOADED:
    import numba


class SymbolicSphericalPotential(SphericalPotential):
    """
    Base class for symbolic spherical potentials.

    Implement a specific spherical density distribution by specifying the (symbolic) density function.
    P.S. Only supports time-independent potential at the moment.
    """

    def __init__(
        self, dens=None, amp=1.0, ro=None, vo=None, amp_units=None, backend="numexpr"
    ):
        """
        Initialize a spherical potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        amp_units : str, optional
            Type of units that amp should have if it has units ('mass', 'velocity2', 'density').

        Notes
        -----
        - 2025-08-10 - Written - Yuzhe Zhang (Uni Mainz)

        """
        SphericalPotential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)
        self.r = sympy.Symbol("r", real=True)
        # self.r = r
        self.dens = dens
        # Compute enclosed mass symbolically
        integrand = 4 * sympy.pi * self.r**2 * self.dens
        M_r = sympy.integrate(integrand, (self.r, 0, self.r))
        self.rawMass = M_r.simplify()
        del integrand

        # Compute the potential symbolically
        Phi_inside_r = -self.rawMass / self.r
        Phi_outside_r = (-4 * sympy.pi) * sympy.integrate(
            self.dens * self.r, (self.r, self.r, sympy.oo)
        ).simplify()
        self.Phi = Phi_inside_r + Phi_outside_r

        # second derivative determined by sympy.diff
        self.d2Phidr2 = sympy.diff(self.Phi, self.r, 2)

        # to accelerate calculation
        # use lambdify + numpy as backend
        if backend == "numpy":
            self.dens = sympy.lambdify(self.r, self.dens, modules="numpy")
            self.Phi = sympy.lambdify(self.r, self.Phi, modules="numpy")
            self.rawMass = sympy.lambdify(self.r, self.rawMass, modules="numpy")
            self.d2Phidr2 = sympy.lambdify(self.r, self.d2Phidr2, modules="numpy")
        # use lambdify + numexpr as backend
        elif backend == "numexpr":
            self.dens = sympy.lambdify(self.r, self.dens, modules="numexpr")
            self.Phi = sympy.lambdify(self.r, self.Phi, modules="numexpr")
            self.rawMass = sympy.lambdify(self.r, self.rawMass, modules="numexpr")
            self.d2Phidr2 = sympy.lambdify(self.r, self.d2Phidr2, modules="numexpr")
        # use numba.njit() + numpy as backend (does not work for _r2deriv when r/a is small)
        elif backend == "numba":
            self.dens = numba.njit()(sympy.lambdify(self.r, self.dens, modules="numpy"))
            self.Phi = numba.njit()(sympy.lambdify(self.r, self.Phi, modules="numpy"))
            self.rawMass = numba.njit()(
                sympy.lambdify(self.r, self.rawMass, modules="numpy")
            )
            self.d2Phidr2 = numba.njit()(
                sympy.lambdify(self.r, self.d2Phidr2, modules="numpy")
            )
        # Creates a compiled numpy ufunc (not work due to values for M_PI (pi) and I (imagnary number))
        # self.dens = ufuncify(self.r, sympy.re(self.dens))
        # self.Phi = ufuncify(self.r, sympy.re(self.Phi))
        # self.rawMass = ufuncify(self.r, sympy.re(self.rawMass))
        # self.d2Phidr2 = ufuncify(self.r, sympy.re(self.d2Phidr2))
        else:
            raise ValueError(f"Unknown backend '{backend}'")

        return None

    def _revaluate(self, r: float, t: float = 0.0):
        """Returns the potential at a given radius r and time t"""
        # return float(self.Phi.evalf(subs={self.r: r}))
        return self.Phi(r)

    def _rforce(self, r: float, t: float = 0.0):
        """Returns the radial force at a given radius r and time t"""
        # return float((-self.rawMass / self.r**2).evalf(subs={self.r: r}))
        return -self.rawMass(r) / r**2

    def _r2deriv(self, r: float, t: float = 0.0):
        """Returns the second radial derivative of the potential at a given radius r and time t"""
        # use the d2Phidr2 obtained by sympy.diff
        # return float(self.d2Phidr2.evalf(subs={self.r: r}))
        return self.d2Phidr2(r)

    def _rdens(self, r: float, t: float = 0.0):
        """Returns the density at a given radius r and time t"""
        # return float(self.dens.evalf(subs={self.r: r}))
        return self.dens(r)
