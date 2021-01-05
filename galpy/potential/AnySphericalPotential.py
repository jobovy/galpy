###############################################################################
#   AnySphericalPotential: Potential of an arbitrary spherical density
###############################################################################
import numpy
from scipy import integrate
from .SphericalPotential import SphericalPotential
class AnySphericalPotential(SphericalPotential):
    """Class that implements the potential of an arbitrary spherical density distribution :math:`\\rho(r)`"""
    def __init__(self,dens=lambda r: 0.64/r/(1+r)**3,amp=1.,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize the potential of an arbitrary spherical density distribution

        INPUT:

           dens= (0.64/r/(1+r)**3) function of a single variable that gives the density as a function of radius (can return a Quantity)

           amp= (1.) amplitude to be applied to the potential

           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2021-01-05 - Written - Bovy (UofT)

        """
        SphericalPotential.__init__(self,amp=amp,ro=ro,vo=vo)
        self._rawdens= dens
        self._rawmass= lambda r: 4.*numpy.pi\
            *integrate.quad(lambda a: a**2*self._rawdens(a),0,r)[0]
        # The potential at zero, try to figure out whether it's finite
        _zero_msg= integrate.quad(lambda a: a*self._rawdens(a),
                                  0,numpy.inf,full_output=True)[-1]
        _infpotzero= 'divergent' in _zero_msg or 'maximum number' in _zero_msg
        self._pot_zero= -numpy.inf if _infpotzero \
            else -4.*numpy.pi\
                 *integrate.quad(lambda a: a*self._rawdens(a),0,numpy.inf)[0]
        # The potential at infinity
        _infmass= 'divergent' \
            in integrate.quad(lambda a: a**2.*self._rawdens(a),
                              0,numpy.inf,full_output=True)[-1]
        self._pot_inf= 0. if not _infmass else numpy.inf
        # Normalize?
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover 
            self.normalize(normalize)
        return None

    def _revaluate(self,r,t=0.):
        """Potential as a function of r and time"""
        if r == 0:
            return self._pot_zero
        elif numpy.isinf(r):
            return self._pot_inf
        else:
            return -self._rawmass(r)/r\
                -4.*numpy.pi*integrate.quad(lambda a: self._rawdens(a)*a,
                                            r,numpy.inf)[0]

    def _rforce(self,r,t=0.):
        return -self._rawmass(r)/r**2
    
    def _r2deriv(self,r,t=0.):
        return -2*self._rawmass(r)/r**3.+4.*numpy.pi*self._rawdens(r)

    def _rdens(self,r,t=0.):
        return self._rawdens(r)
