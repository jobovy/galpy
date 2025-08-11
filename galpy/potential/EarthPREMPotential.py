###############################################################################
#   EarthPREMPotentialpy: The potential of Earth
###############################################################################
import numpy

from ..util import conversion
from ..util._optional_deps import _SYMPY_LOADED

# from .Potential import Potential
from .SymbolicSphericalPotential import SymbolicSphericalPotential

if _SYMPY_LOADED:
    import sympy
    from sympy import symbols, integrate, Piecewise, pi, latex, sqrt


r = sympy.Symbol("r", real=True, positive=True)
EARTH_RADIUS_KM = 6371.0
pieces = [
        (13.0885 - 8.8381 * (r / EARTH_RADIUS_KM) ** 2, 0, 1221.5),
        (12.5815 - 1.2638 * (r / EARTH_RADIUS_KM)
         - 3.6426 * (r / EARTH_RADIUS_KM) ** 2
         - 5.5281 * (r / EARTH_RADIUS_KM) ** 3, 1221.5, 3480.0),
        (7.9565 - 6.4761 * (r / EARTH_RADIUS_KM)
         + 5.5283 * (r / EARTH_RADIUS_KM) ** 2
         - 3.0807 * (r / EARTH_RADIUS_KM) ** 3, 3480.0, 5701.0),
        (5.3197 - 1.4836 * (r / EARTH_RADIUS_KM), 5701.0, 5771.0),
        (11.2494 - 8.0298 * (r / EARTH_RADIUS_KM), 5771.0, 5971.0),
        (7.1089 - 3.8045 * (r / EARTH_RADIUS_KM), 5971.0, 6151.0),
        (2.6910 + 0.6924 * (r / EARTH_RADIUS_KM), 6151.0, 6346.6),
        (2.6910 + 0.6924 * (r / EARTH_RADIUS_KM), 6346.6, 6356.0),
        (2.6, 6356.0, EARTH_RADIUS_KM),
        # Explicitly include density for r >= EARTH_RADIUS_KM:
        (0.0, EARTH_RADIUS_KM, sympy.oo),
    ]

def earth_PREM_density_sym():
    r = sympy.Symbol("r", real=True, positive=True)
    # Build the Piecewise arguments from the list
    pw_args = []
    for expr, rmin, rmax in pieces:
        cond = (r >= rmin) & (r < rmax)
        pw_args.append((expr, cond))

    # Add fallback condition for r >= EARTH_RADIUS_KM (density = 0)
    pw_args.append((0, True))

    dens_sym = sympy.Piecewise(*pw_args)
    return dens_sym, r


class EarthPREMPotential(SymbolicSphericalPotential):
    r"""Class that implements the Earth potential following the Preliminary reference Earth model (PREM). 
    The potential is given by
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

    def __init__(self, amp=1.0, R=None, normalize=False, ro=None, vo=None):
        """
        Initialize the gravitational potential of earth.

        Parameters
        ----------
        amp : float or Quantity, optional
            Amplitude to be applied to the potential. Can be a Quantity with units of mass density or Gxmass density.
        R : float or Quantity, optional
            
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2025-08-10 - Written - Yuzhe Zhang (Uni Mainz)
        """
        SymbolicSphericalPotential.__init__(
            self,
            dens_sym=earth_PREM_density_sym,
        )
        # R = conversion.parse_length(R, ro=self._ro)
        # self.R = R
        # self._R2 = self.R**2.0
        # self._R3 = self.R**3.0
        # if normalize or (
        #     isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        # ):  # pragma: no cover
        #     self.normalize(normalize)
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False
