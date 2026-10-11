###############################################################################
#   AdiabaticContractionWrapperPotential.py: Wrapper to adiabatically
#                                            contract a DM halo in response
#                                            to the growth of a baryonic
#                                            component
###############################################################################
import numpy
from scipy import integrate

from ..backend import to_host
from ..backend._namespaces import (
    namespace_from_arrays,
    requires_backend_grad,
    under_trace,
)
from ..backend.interpolate import interp1d_clamped, interp_linear
from ..backend.optimize import fixed_point
from ..util import conversion
from .Force import Force
from .interpSphericalPotential import interpSphericalPotential
from .Potential import _pot_grad_namespace, flatten


# Note: not actually implemented as a WrapperPotential!
class AdiabaticContractionWrapperPotential(interpSphericalPotential):
    """AdiabaticContractionWrapperPotential: Wrapper to adiabatically contract a DM halo in response to the growth of a baryonic component. Use for example as::

        dm= AdiabaticContractionWrapperPotential(
            pot=MWPotential2014[2],
            baryonpot=MWPotential2014[:2]
        )

    to contract the dark-matter halo in ``MWPotential2014`` according to the baryon distribution within it. The basic physics of the adiabatic contraction is that a fraction ``f_bar`` of the mass in the original potential ``pot`` cools adiabatically to form a baryonic component ``baryonpot``; this wrapper computes the resulting dark-matter potential using different approximations in the literature.

    """

    def __init__(
        self,
        amp=1.0,
        pot=None,
        baryonpot=None,
        method="cautun",
        f_bar=0.157,
        rmin=None,
        rmax=50.0,
        ro=None,
        vo=None,
    ):
        """
        Initialize a AdiabaticContractionWrapper Potential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1.).
        pot : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            Representing the density that is adiabatically contracted.
        baryonpot : Potential instance or a combined potential formed using addition (pot1+pot2+…), optional
            Representing the density of baryons whose growth causes the contraction.
        method : {'cautun', 'blumenthal', 'gnedin'}, optional
            Type of adiabatic-contraction formula:

            - 'cautun' for that from Cautun et al. 2020 [1]_;
            - 'blumenthal' for that from Blumenthal et al. 1986 [2]_;
            - 'gnedin' for that from Gnedin et al. 2004 [3]_.

            (default: 'cautun')
        f_bar : float, optional
            Universal baryon fraction; if None, calculated from pot and baryonpot assuming that at rmax the halo contains the universal baryon fraction; leave this at the default value unless you know what you are doing (default: 0.157).
        rmin : float, optional
            Minimum radius to consider (default: rmax/2500; don't set this to zero).
        rmax : float or Quantity, optional
            Maximum radius to consider (default: 50.).
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2021-03-21 - Started based on Marius Cautun's code - Bovy (UofT)

        References
        ----------
        .. [1] Cautun, M et al. (2020), Mon. Not. Roy. Astron. Soc., 494, 4291. ADS:  https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4291C
        .. [2] Blumenthal et al. (1986), Astrophys. J., 301, 27. ADS:  https://ui.adsabs.harvard.edu/abs/1986ApJ...301...27B
        .. [3] Gnedin et al. (2004), Astrophys. J., 616, 16. ADS:  https://ui.adsabs.harvard.edu/abs/2004ApJ...616...16G
        """
        # Initialize with Force just to parse (ro,vo)
        Force.__init__(self, ro=ro, vo=vo)
        rmax = conversion.parse_length(rmax, ro=self._ro)
        rmin = (
            conversion.parse_length(rmin, ro=self._ro)
            if not rmin is None
            else rmax / 2500.0
        )
        # Compute baryon and DM enclosed masses on radial grid
        from ..potential import mass

        # A differentiated parameter (of pot, baryonpot, or f_bar) builds the
        # profile in its namespace, so the contracted halo carries d/d(param)
        xp = _pot_grad_namespace(flatten([pot, baryonpot]))
        for v in (f_bar, amp):
            if xp is None and (under_trace(v) or requires_backend_grad(v)):
                xp = namespace_from_arrays((v,))
        rgrid = numpy.geomspace(rmin, rmax, 301)
        if xp is None:
            # numpy/scipy construction: host the (forced-backend) masses
            baryon_mass = numpy.array(
                to_host([mass(baryonpot, r, use_physical=False) for r in rgrid])
            )
            dm_mass = numpy.array(
                to_host([mass(pot, r, use_physical=False) for r in rgrid])
            )
        else:
            rgrid = xp.asarray(rgrid)
            baryon_mass = mass(baryonpot, rgrid, use_physical=False)
            dm_mass = mass(pot, rgrid, use_physical=False)
        # Adiabatic contraction
        if f_bar is None:
            f_bar = baryon_mass[-1] / (baryon_mass[-1] + dm_mass[-1])
        if method.lower() == "cautun":
            new_rforce = _contraction_Cautun2020(rgrid, dm_mass, baryon_mass, f_bar)
        elif method.lower() == "gnedin":
            new_rforce = _contraction_Gnedin2004(
                rgrid,
                dm_mass,
                baryon_mass,
                pot.rvir(overdens=180.0, wrtcrit=False),
                f_bar,
            )
        elif method.lower() == "blumenthal":
            new_rforce = _contraction_Blumenthal1986(rgrid, dm_mass, baryon_mass, f_bar)
        else:  # pragma: no cover
            raise ValueError(f"Adiabatic contraction method '{method}' not recognized")
        if xp is None:
            # Add central point
            rgrid = numpy.concatenate(([0.0], rgrid))
            new_rforce = numpy.concatenate(([0.0], new_rforce))
            new_rforce_func = lambda r: -numpy.interp(r, rgrid, new_rforce)
            # Potential at zero = int_0^inf dr rforce, and enc. mass constant
            # outside of last rgrid point
            Phi0 = (
                integrate.quad(new_rforce_func, rgrid[0], rgrid[-1])[0]
                - new_rforce[-1] * rgrid[-1]
            )
        else:
            zero = xp.zeros_like(rgrid[:1])
            rgrid = xp.concat((zero, rgrid))
            new_rforce = xp.concat((zero, new_rforce))
            new_rforce_func = lambda r: -interp_linear(xp, rgrid, new_rforce, r)
            # the force is piecewise linear, so its integral is the exact
            # trapezoid sum (scipy's quad above stops at 50 subintervals, ~1e-6)
            Phi0 = (
                -xp.sum(
                    0.5 * (new_rforce[1:] + new_rforce[:-1]) * (rgrid[1:] - rgrid[:-1])
                )
                - new_rforce[-1] * rgrid[-1]
            )
        interpSphericalPotential.__init__(
            self, rforce=new_rforce_func, rgrid=rgrid, Phi0=Phi0, ro=ro, vo=vo
        )
        if xp is not None:  # honour amp (the numpy path drops it: fixed separately)
            self._amp = amp


# Parameters go through fixed_point's args, not a closure: its backend
# iteration detaches them and differentiates implicitly.
def _contraction_Cautun2020(r, M_DMO, Mbar, fbar):
    # solve for the contracted enclosed DM mass
    func_M_DM_contract = lambda M, M_DMO, Mbar, fbar: (
        M_DMO * 1.023 * (M_DMO / (1.0 - fbar) / (M + Mbar)) ** -0.54
    )
    M_DM = fixed_point(func_M_DM_contract, M_DMO, args=(M_DMO, Mbar, fbar))
    return M_DM / M_DMO * M_DMO / r**2.0


def _contraction_Blumenthal1986(r, M_DMO, Mbar, fbar):
    # solve for the contracted radius 'rf' containing the same DM mass
    # as enclosed for r
    func_r_contract = lambda rf, r, M_DMO, Mbar, fbar: (
        r * (M_DMO / (1.0 - fbar)) / (M_DMO + interp1d_clamped(r, Mbar)(rf))
    )
    rf = fixed_point(func_r_contract, r, args=(r, M_DMO, Mbar, fbar))
    # now find how much the enclosed mass increased at r
    func_M_DM = interp1d_clamped(rf, M_DMO)
    return func_M_DM(r) / r**2.0


def _contraction_Gnedin2004(r, M_DMO, M_bar, Rvir, fbar):
    # solve for the contracted radius 'rf' containing the same DM mass
    # as enclosed for r
    A, w = 0.85, 0.8
    func_r_mean = lambda ri, Rvir: A * Rvir * (ri / Rvir) ** w
    M_DMO_rmean = interp1d_clamped(r, M_DMO)(func_r_mean(r, Rvir))
    func_r_contract = lambda rf, r, M_DMO_rmean, M_bar, Rvir, fbar: (
        r
        * (M_DMO_rmean / (1.0 - fbar))
        / (M_DMO_rmean + interp1d_clamped(r, M_bar)(func_r_mean(rf, Rvir)))
    )
    rf = fixed_point(func_r_contract, r, args=(r, M_DMO_rmean, M_bar, Rvir, fbar))
    # now find how much the enclosed mass increased at r
    func_M_DM = interp1d_clamped(rf, M_DMO)
    return func_M_DM(r) / r**2.0
