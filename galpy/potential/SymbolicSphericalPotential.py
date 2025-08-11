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

# def sym_to_num(expr, r, r_val, t=sympy.Symbol("t", real=True), t_val:float=0.0):
#         expr_sub = expr.subs(r, r_val)
#         expr_no_min = expr_sub.replace(
#             lambda x: isinstance(x, sympy.Min),
#             lambda x: min(*[arg.evalf() for arg in x.args])
#         )
#         return float(expr_no_min.evalf())


class SymbolicSphericalPotential(Potential):
    """Base class for symbolic spherical potentials.

    Implement a specific spherical density distribution by specifying the (symbolic) density function.

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
        - 2025-08-10 - Written - Yuzhe Zhang (Uni Mainz)

        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)

        self.dens_sym, self.r = dens_sym()
        # Compute enclosed mass symbolically
        integrand = 4 * sympy.pi * self.r**2 * self.dens_sym
        M_r = sympy.integrate(integrand, (self.r, 0, self.r))
        self.rawMass_sym = M_r.simplify()
        del integrand

        # Compute the potential symbolically
        self.Phi = -self.rawMass_sym / self.r

        # second derivative determined by sympy.diff
        self.d2Phidr2 = sympy.diff(self.Phi, self.r, 2)

        return None

    def _revaluate(self, r: float, t: float = 0.0):
        """Returns the potential at a given radius r and time t"""
        return float(self.Phi.evalf(subs={self.r: r}))

    def _rforce(self, r: float, t: float = 0.0):
        """Returns the radial force at a given radius r and time t"""
        return float((-self.rawMass_sym / self.r**2).evalf(subs={self.r: r}))

    def _r2deriv(self, r: float, t: float = 0.0):
        """Returns the second radial derivative of the potential at a given radius r and time t"""
        # obtained analytically from the relation
        # if    dΦ/dr = M(r) / r
        # then  d²Φ/dr² = M''(r) / r - 2 M' / r² + 2 M(r) / r³
        #       d²Φ/dr² = M''(r) / r - 4π ρ(r) + 2 M(r) / r³
        # using the enclosed mass M(r) and density ρ(r) directly.
        # expr = (
        #     -sympy.diff(pot.rawMass_sym, pot.r, 2) / pot.r
        #     + 2.0 * sympy.diff(pot.rawMass_sym, pot.r, 1) / pot.r**2.0
        #     - 2 * pot.rawMass_sym / pot.r**3.0
        # )
        return float(self.d2Phidr2.evalf(subs={self.r: r}))

    def _rdens(self, r: float, t: float = 0.0):
        """Returns the density at a given radius r and time t"""
        return float(self.dens_sym.evalf(subs={self.r: r}))

    def _rmass(self, r: float, t: float = 0.0):
        """Returns the density at a given radius r and time t"""
        return float(self.rawMass_sym.evalf(subs={self.r: r}))

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        """Find potential at (R, z, phi, t)"""
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

    def _mass(self, R, z, phi=0.0, t=0.0):
        R, z = numpy.float64(R), numpy.float64(z)  # Avoid indexing issues
        r = numpy.sqrt(R**2.0 + z**2.0)
        return self._rmass(r=r)

    def _rforce_sym(self, r, t=0.0):
        return -self._rawmass_sym.evalf(subs={self.r: r}) / r**2

    def _r2deriv_sym(self, r, t=0.0):
        # Directly compute the second derivative
        d2Phi_dr2 = sympy.diff(self.Phi, self.r, 2)
        return d2Phi_dr2.evalf(subs={self.r: r})
