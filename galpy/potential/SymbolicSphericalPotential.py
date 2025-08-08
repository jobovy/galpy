###############################################################################
#   SphericalPotential.py: base class for potentials corresponding to
#                          spherical density profiles
###############################################################################
import numpy
from scipy import integrate

from .Potential import Potential
from ..util._optional_deps import _SYMPY_LOADED

if _SYMPY_LOADED:
    import sympy
    from sympy import symbols, integrate, Piecewise, pi, latex, sqrt


class SymbolicSphericalPotential(Potential):
    """Base class for symbolic spherical potentials.

    Implement a specific spherical density distribution by specifying the density function.
    """

    def __init__(self, dens_sym=None, amp=1.0, ro=None, vo=None, amp_units=None):
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
        - 2025-07-23 - Written - Yuzhe Zhang (Uni Mainz)

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)
        self.r_sym = sympy.Symbol("r_sym")
        self.dens_sym = dens_sym
        # Compute enclosed mass symbolically
        integrand = 4 * sympy.pi * self.r_sym**2 * self.dens_sym
        M_r = sympy.integrate(integrand, (self.r_sym, 0, self.r_sym), conds="none")
        self.rawMass_sym = M_r.simplify()
        del integrand

        # get the potential
        # r_prime = sympy.symbols('r_prime', real=True, positive=True)
        integrand = self.rawMass_sym / self.r_sym**2
        self.Phi = sympy.integrate(integrand, (self.r_sym, self.r_sym, 0))
        del integrand

        return None

    def _rdens(self, r, t=0.0):
        """Returns the density at a given radius r (in km)"""
        density_expr = self.dens_sym.subs({self.r_sym: r})

        # Evaluate the expression to get a numerical value
        return float(density_expr)

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._revaluate(r, t=t)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._rforce(r, t=t) * R / r

    def _zforce(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._rforce(r, t=t) * z / r

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return (
            self._r2deriv(r, t=t) * R**2.0 / r**2.0
            - self._rforce(r, t=t) * z**2.0 / r**3.0
        )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return (
            self._r2deriv(r, t=t) * z**2.0 / r**2.0
            - self._rforce(r, t=t) * R**2.0 / r**3.0
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return (
            self._r2deriv(r, t=t) * R * z / r**2.0
            + self._rforce(r, t=t) * R * z / r**3.0
        )

    def _dens(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._rdens(r, t=t)

    def _mass(self, R, z=None, t=0.0):
        if z is not None:
            raise AttributeError  # use general implementation
        R = numpy.float64(R)  # Avoid indexing issues
        return self.rawMass_sym.subs({self.r_sym: R})

    def _rforce_sym(self, r, t=0.0):
        return -self._rawmass_sym.evalf(subs={self.r_sym: r}) / r**2

    def _r2deriv_sym(self, r, t=0.0):
        # Directly compute the second derivative
        d2Phi_dr2 = sympy.diff(self.Phi, self.r_sym, 2)
        return d2Phi_dr2.evalf(subs={self.r_sym: r})
