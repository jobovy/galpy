###############################################################################
#   KingPotential.py: Potential of a King profile
###############################################################################
import numpy
from .interpSphericalPotential import interpSphericalPotential
class KingPotential(interpSphericalPotential):
    """KingPotential.py: Potential of a King profile, defined from the distribution function

    .. math::

      f(\\mathcal{E}) = \begin{cases} \\rho_1\\,(2\\pi\\sigma^2)^{-3/2}\\,\\left(e^{\\mathcal{E}/\\sigma^2}-1\\right), & \mathcal{E} > 0\\
0, & \mathcal{E} \leq 0\end{cases}

    where :math:`\mathcal{E}` is the binding energy.
    """
    def __init__(self,W0=2.,M=3.,rt=1.5,npt=1001,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a King potential

        INPUT:

           W0= (2.) dimensionless central potential W0 = Psi(0)/sigma^2 (in practice, needs to be <~ 200, where the DF is essentially isothermal)

           M= (1.) total mass (can be a Quantity)

           rt= (1.) tidal radius (can be a Quantity)

           npt= (1001) number of points to use to solve for Psi(r) when solving the King DF

           scfa= (1.) scale parameter used in the SCF representation of the potential

           scfN= (30) number of expansion coefficients to use in the SCF representation of the potential     

           ro=, vo= standard galpy unit scaling parameters

        OUTPUT:

           (none; sets up instance)

        HISTORY:

           2020-07-11 - Written - Bovy (UofT)

        """
        # Set up King DF
        from ..df.kingdf import kingdf
        kdf= kingdf(W0,M=M,rt=rt,ro=ro,vo=vo)
        interpSphericalPotential.__init__(\
            self,
            rforce=lambda r: kdf._mass_scale/kdf._radius_scale**2.
                            *numpy.interp(r/kdf._radius_scale,
                                          kdf._scalefree_kdf._r,
                                          kdf._scalefree_kdf._dWdr),
            rgrid=kdf._scalefree_kdf._r*kdf._radius_scale,
            Phi0=-kdf.W0*kdf._mass_scale/kdf._radius_scale,
            ro=ro,vo=vo)
