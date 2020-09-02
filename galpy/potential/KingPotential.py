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
    def __init__(self,W0=2.,M=3.,rt=1.5,npt=1001,_sfkdf=None,ro=None,vo=None):
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

           ro=, vo= standard galpy unit scaling parameters

        OUTPUT:

           (none; sets up instance)

        HISTORY:

           2020-07-11 - Written - Bovy (UofT)

        """
        # Set up King DF
        if _sfkdf is None:
            from ..df.kingdf import _scalefreekingdf
            sfkdf= _scalefreekingdf(W0)
            sfkdf.solve(npt)
        else:
            sfkdf= _sfkdf
        mass_scale= M/sfkdf.mass
        radius_scale= rt/sfkdf.rt
        interpSphericalPotential.__init__(\
            self,
            rforce=lambda r: mass_scale/radius_scale**2.
                            *numpy.interp(r/radius_scale,
                                          sfkdf._r,
                                          sfkdf._dWdr),
            rgrid=sfkdf._r*radius_scale,
            Phi0=-W0*mass_scale/radius_scale,
            ro=ro,vo=vo)
