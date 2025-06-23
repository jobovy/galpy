###############################################################################
#   ?????????.py: The potential of Earth
###############################################################################
import numpy

from ..util import conversion
from ..util._optional_deps import _SYMPY_LOADED
# from .Potential import Potential
from .AnySphericalPotential import AnySphericalPotential
# from .AnySphericalPotential_Symbolic import AnySphericalPotential_Symbolic
if _SYMPY_LOADED:
    import sympy
    from sympy import symbols, integrate, Piecewise, pi, latex , sqrt

def earth_PREM_density(radius_km):
    # Use the coefficients of the polynomials describing the Preliminary Reference Earth Model (PREM) to find out density.
    radius_km = numpy.abs(radius_km)

    # Earth's radius in km
    EARTH_RADIUS_KM = 6371

    # PREM density functions for each region (x = r/6371)
    def density_inner_core(x):
        return 13.0885 - 8.8381 * x**2

    def density_outer_core(x):
        return 12.5815 - 1.2638 * x - 3.6426 * x**2 - 5.5281 * x**3

    def density_lower_mantle(x):
        return 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3

    def density_transition_zone(x):
        if x <= 5771 / EARTH_RADIUS_KM:
            return 5.3197 - 1.4836 * x
        elif x <= 5971 / EARTH_RADIUS_KM:
            return 11.2494 - 8.0298 * x
        else:
            return 7.1089 - 3.8045 * x

    def density_lvz_lid(x):
        return 2.6910 + 0.6924 * x

    def density_crust(x):
        if x <= 6356 / EARTH_RADIUS_KM:
            return 2.900
        else:
            return 2.600

    def density_ocean(x):
        return 1.020

    # # Generate radius values (in km) at 10 km intervals
    # radius_km = np.arange(0, EARTH_RADIUS_KM., 10)

    # Calculate density for each radius
    # density = []
    x = radius_km / EARTH_RADIUS_KM
    # radius_km = radius_km
    if radius_km <= 1221.5:
        return density_inner_core(x)
    elif radius_km <= 3480.0:
        return density_outer_core(x)
    elif radius_km <= 5701.0:
        return density_lower_mantle(x)
    elif radius_km <= 6151.0:
        return density_transition_zone(x)
    elif radius_km <= 6346.6:
        return density_lvz_lid(x)
    elif radius_km <= 6368.0:
        return density_crust(x)
    elif radius_km <= 6371.0:
        return density_ocean(x)
    else: 
        return 0.0

def earth_PREM_density_sym():
    # R_sym, z_sym = sympy.symbols('R z')
    # r2 = R_sym**2.0 + z_sym**2.0
    # r_sym = sympy.sqrt(r2)
    r_sym = sympy.Symbol('r', real=True, positive=True)
    # R = sympy.symbols('r', real=True, positive=True)
    EARTH_RADIUS_KM = 6371.0  # Earth's radius in km
    x = r_sym / EARTH_RADIUS_KM
 
    dens_sym = sympy.Piecewise(
        (13.0885 - 8.8381*x**2, r_sym < 1221.5),
        (12.5815 - 1.2638*x - 3.6426*x**2 - 5.5281*x**3, (r_sym >= 1221.5) & (r_sym < 3480.0)),
        # (7.9565 - 6.4761*x + 5.5283*x**2 - 3.0807*x**3, (r_sym >= 3480.0) & (r_sym < 5701.0)),
        # (5.3197 - 1.4836*x, (r_sym >= 5701.0) & (r_sym < 5771.0)),
        # (11.2494 - 8.0298*x, (r_sym >= 5771.0) & (r_sym < 5971.0)),
        # (7.1089 - 3.8045*x, (r_sym >= 5971.0) & (r_sym < 6151.0)),
        # (2.6910 + 0.6924*x, (r_sym >= 6151.0) & (r_sym < 6346.6)),
        # (2.6910 + 0.6924*x, (r_sym >= 6346.6) & (r_sym < 6356.0)),
        # (2.6, r_sym >= 6356.0),
        (0, True)  # for r > 6371.0 and safety
    )
    return dens_sym, r_sym

class EarthPREMPotential(AnySphericalPotential):
    r"""Class that implements the Earth potential following the Preliminary reference Earth model (PREM). 
    The potential is given by

    .. math::

        \[
            \rho(r) =
            \begin{cases}
            13.0885 - 8.8381 x^2, & 0 \le r < 1221.5 \quad \text{(Inner Core)} \\
            12.5815 - 1.2638 x - 3.6426 x^2 - 5.5281 x^3, & 1221.5 \le r < 3480.0 \quad \text{(Outer Core)} \\
            7.9565 - 6.4761 x + 5.5283 x^2 - 3.0807 x^3, & 3480.0 \le r < 5701.0 \quad \text{(Lower Mantle)} \\
            5.3197 - 1.4836 x, & 5701.0 \le r < 5771.0 \quad \text{(Upper Transition Zone)} \\
            11.2494 - 8.0298 x, & 5771.0 \le r < 5971.0 \quad \text{(Mid Transition Zone)} \\
            7.1089 - 3.8045 x, & 5971.0 \le r < 6151.0 \quad \text{(Lower Transition Zone)} \\
            2.6910 + 0.6924 x, & 6151.0 \le r < 6291.0 \quad \text{(Low-Velocity Zone)} \\
            2.6910 + 0.6924 x, & 6291.0 \le r < 6346.6 \quad \text{(Lithospheric Mantle)} \\
            2.900, & 6346.6 \le r \le 6356.0 \quad \text{(Crust)} \\
            2.600, & 6356.0 \le r \le 6368.0 \quad \text{(Crust)} \\
            1.020, & 6368.0 \le r \le 6371.0 \quad \text{(Ocean)}
            \end{cases}
        \]

    
    """

    def __init__(self, amp=1.0, R=6371e3, normalize=False, ro=None, vo=None):
        """
        Initialize ..................................................................

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential. Can be a Quantity with units of mass density or Gxmass density.
        R : float or Quantity, optional
            Earth radius 6,371,000 m. 
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2019-12-20 - Written - Bovy (UofT)
        """

        AnySphericalPotential.__init__(self, amp=amp, dens=earth_PREM_density, normalize=normalize,ro=ro, vo=vo)
        R = conversion.parse_length(R, ro=self._ro)
        self.R = R
        self._R2 = self.R**2.0
        self._R3 = self.R**3.0
        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.dens_sym, self.r_sym = earth_PREM_density_sym()  # , self.R_sym, self.z_sym
        # Compute enclosed mass symbolically
        integrand = 4 * sympy.pi * self.r_sym**2 * self.dens_sym
        M_r = sympy.integrate(integrand, (self.r_sym, 0, self.r_sym), conds='none')
        self._rawmass_sym = M_r.simplify()
        # get the potential
        # r_prime = sympy.symbols('r_prime', real=True, positive=True)
        integrand = self._rawmass_sym / self.r_sym**2
        self.Phi = sympy.integrate(integrand, (self.r_sym, self.r_sym, 0))

        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return r2 - 3.0 * self._R2
        else:
            return -2.0 * self._R3 / numpy.sqrt(r2)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return -2.0 * R
        else:
            return -2.0 * self._R3 * R / r2**1.5

    def _zforce(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return -2.0 * z
        else:
            return -2.0 * self._R3 * z / r2**1.5

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return 2.0
        else:
            return 2.0 * self._R3 / r2**1.5 - 6.0 * self._R3 * R**2.0 / r2**2.5

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return 2.0
        else:
            return 2.0 * self._R3 / r2**1.5 - 6.0 * self._R3 * z**2.0 / r2**2.5

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return 0.0
        else:
            return -6.0 * self._R3 * R * z / r2**2.5

    def _dens(self, R, z, phi=0.0, t=0.0):
        r2 = R**2.0 + z**2.0
        if r2 < self._R2:
            return 1.5 / numpy.pi
        else:
            return 0.0
        
    def _rforce_sym(self, r, t=0.0):
        return -self._rawmass_sym.evalf(subs={self.r_sym: r}) / r**2

    def _r2deriv_sym(self, r, t=0.0):
        # Directly compute the second derivative
        d2Phi_dr2 = sympy.diff(self.Phi, self.r_sym, 2)
        return d2Phi_dr2.evalf(subs={self.r_sym: r})


# if __name__ == "__main__":
#     # Usage
#     rho, R = earth_PREM_density_sym()

#     # Print in readable format
#     print(rho)

#     # Or, print LaTeX version (e.g. for inserting into a document)
#     print(latex(rho))