# Class that implements anisotropic DFs of the Osipkov-Merritt type
import numpy
from scipy import integrate, special
from ..util import conversion
from ..potential import evaluatePotentials
from .sphericaldf import anisotropicsphericaldf

class osipkovmerrittdf(anisotropicsphericaldf):
    """Class that implements anisotropic DFs of the Osipkov-Merritt type with radially varying anisotropy
    
    .. math::

        \\beta(r) = \\frac{1}{1+r_a^2/r^2}

    with :math:`r_a` the anistropy radius.

    """
    def __init__(self,pot=None,ra=1.4,scale=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize a DF with Osipkov-Merritt anisotropy

        INPUT:

            pot - Hernquist potential which determines the DF

            ra - anisotropy radius (can be a Quantity)

            scale - Characteristic scale radius to aid sampling calculations. 
                Not necessary, and will also be overridden by value from pot 
                if available.

        OUTPUT:

            None

        HISTORY:

            2020-11-12 - Written - Bovy (UofT)

        """
        anisotropicsphericaldf.__init__(self,pot=pot,scale=scale,ro=ro,vo=vo)
        self._ra= -conversion.parse_length(ra,ro=self._ro)
        self._ra2= self._ra**2.

    def _call_internal(self,*args):
        """
        NAME:

            _call_internal

        PURPOSE:

            Evaluate the DF for an Osipkov-Merritt-anisotropy DF

        INPUT:

            E - The energy

            L - The angular momentum

        OUTPUT:

            fH - The value of the DF

        HISTORY:

            2020-11-12 - Written - Bovy (UofT)

        """
        E, L, _= args
        return self.fQ(-E-0.5*L**2./self._ra2)

    def _sample_eta(self,r,n=1):
        """Sample the angle eta which defines radial vs tangential velocities"""
        # cumulative distribution of x = cos eta satisfies
        # x/(sqrt(A+1 -A* x^2)) = 2 b - 1 = c
        # where b \in [0,1] and A = (r/ra)^2
        # Solved by
        # x = c sqrt(1+[r/ra]^2) / sqrt( [r/ra]^2 c^2 + 1 ) for c > 0 [b > 0.5]
        # and symmetric wrt c
        c= numpy.random.uniform(size=n)
        x= c*numpy.sqrt(1+r**2./self._ra2)/numpy.sqrt(r**2./self._ra2*c**2.+1)
        x*= numpy.random.choice([1.,-1.],size=n)
        return numpy.arccos(x)

    def _p_v_at_r(self,v,r):
        """p( v*sqrt[1+r^2/ra^2*sin^2eta] | r) used in sampling """
        return self.fQ(-evaluatePotentials(self._pot,r,0,use_physical=False)\
                       -0.5*v**2.)*v**2.

    def _sample_v(self,r,eta,n=1):
        """Generate velocity samples"""
        # Use super-class method to obtain v*[1+r^2/ra^2*sin^2eta]
        out= super(osipkovmerrittdf,self)._sample_v(r,eta,n=n)
        # Transform to v
        return out/numpy.sqrt(1.+r**2./self._ra2*numpy.sin(eta)**2.)

    def _vmomentdensity(self,r,n,m):
         if m%2 == 1 or n%2 == 1:
             return 0.
         psir= -evaluatePotentials(self._pot,r,0,use_physical=False)
         return 2.*numpy.pi*integrate.quad(lambda v: v**(2.+m+n)
                                    *self.fQ(-evaluatePotentials(self._pot,r,0,
                                                         use_physical=False)
                                             -0.5*v**2.),
                             0.,self._vmax_at_r(self._pot,r))[0]\
            *special.gamma(m/2.+1.)*special.gamma((n+1)/2.)/\
            special.gamma(0.5*(m+n+3.))/(1+r**2./self._ra2)**(m/2+1)
    
