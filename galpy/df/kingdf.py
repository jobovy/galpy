# Class that represents a King DF
import numpy
from scipy import special, integrate, interpolate
from .sphericaldf import sphericaldf

_FOURPI= 4.*numpy.pi
_TWOOVERSQRTPI= 2./numpy.sqrt(numpy.pi)

class kingdf(sphericaldf):
    """Class that represents a King DF"""
    def __init__(self,W0,M=1.,rt=1.,npt=1001,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a King DF

        INPUT:

           W0 - dimensionless central potential W0 = Psi(0)/sigma^2 (in practice, needs to be <~ 200, where the DF is essentially isothermal)

           M= (1.) total mass (can be a Quantity)

           rt= (1.) tidal radius (can be a Quantity)

           npt= (1001) number of points to use to solve for Psi(r)

           ro=, vo= standard galpy unit scaling parameters

        OUTPUT:

           (none; sets up instance)

        HISTORY:

           2020-07-09 - Written - Bovy (UofT)

        """
        sphericaldf.__init__(self,ro=ro,vo=vo)
        # Need to add parsing of Quantity inputs...
        
        self.W0= W0
        # Solve (mass,rtidal)-scale-free model, which is the basis for
        # the full solution
        self._scalefree_kdf= _scalefreekingdf(self.W0)
        self._scalefree_kdf.solve(npt)
        # Set up scaling factors
        self._radius_scale= rt/self._scalefree_kdf.rt
        self._mass_scale= M/self._scalefree_kdf.mass
        self._velocity_scale= numpy.sqrt(self._mass_scale/self._radius_scale)
        self._density_scale= self._mass_scale/self._radius_scale**3.
        # Store central density, r0...
        self.rho0= self._scalefree_kdf.rho0*self._density_scale
        self.r0= self._scalefree_kdf.r0*self._radius_scale
        self.c= self._scalefree_kdf.c # invariant
        self.rt= rt # for convenience

    def dens(self,r):
        return self._scalefree_kdf.dens(r/self._radius_scale)\
            *self._density_scale
        
class _scalefreekingdf(object):
    """Internal helper class to solve the scale-free King DF model, that is, the one that only depends on W = Psi/sigma^2"""
    def __init__(self,W0):
        self.W0= W0

    def solve(self,npt=1001):
        """Solve the model W(r) at npt points (note: not equally spaced in 
        either r or W, because combination of two ODEs for different r ranges)"""
        # Set up arrays for outputs
        r= numpy.zeros(npt)
        W= numpy.zeros(npt)
        dWdr= numpy.zeros(npt)
        # Initialize (r[0]=0 already)
        W[0]= self.W0
        # Determine central density and r0
        self.rho0= self._dens_W(self.W0)
        self.r0= numpy.sqrt(9./4./numpy.pi/self.rho0)
        # First solve Poisson equation ODE from r=0 to r0 using form
        # d^2 Psi / dr^2 =  ... (d psi / dr = v, r^2 dv / dr = RHS-2*r*v)
        if self.W0 < 2.:
            rbreak= self.r0/100.
        else:
            rbreak= self.r0
        #Using linspace focuses on what happens ~rbreak rather than on <<rbreak
        # which is what you want, because W ~ constant at r <~ r0
        r[:npt//2]= numpy.linspace(0.,rbreak,npt//2)
        sol= integrate.solve_ivp(\
                lambda t,y: [y[1],-_FOURPI*self._dens_W(y[0])
                             -(2.*y[1]/t if t > 0. else 0.)],
                [0.,rbreak],[self.W0,0.],method='LSODA',t_eval=r[:npt//2])
        W[:npt//2]= sol.y[0]
        dWdr[:npt//2]= sol.y[1]
        # Then solve Poisson equation ODE from Psi(r0) to Psi=0 using form
        # d^2 r / d Psi^2 = ... (d r / d psi = 1/v, dv / dpsi = 1/v(RHS-2*r*v))
        # Added advantage that this becomes ~log-spaced in r, which is what
        # you want
        W[npt//2-1:]= numpy.linspace(sol.y[0,-1],0.,npt-npt//2+1)
        sol= integrate.solve_ivp(\
                lambda t,y: [1./y[1],
                             -1./y[1]*(_FOURPI*self._dens_W(t)
                                       +2.*y[1]/y[0])],
                [sol.y[0,-1],0.],[rbreak,sol.y[1,-1]],
                method='LSODA',t_eval=W[npt//2-1:])
        r[npt//2-1:]= sol.y[0]
        dWdr[npt//2-1:]= sol.y[1]
        # Store solution
        self._r= r
        self._W= W
        self._dWdr= dWdr
        # Also store density at these points, and the tidal radius
        self._rho= self._dens_W(self._W)
        self.rt= r[-1]
        self.c= numpy.log10(self.rt/self.r0)
        # Interpolate solution
        self._W_from_r=\
            interpolate.InterpolatedUnivariateSpline(self._r,self._W,k=3)
        # Compute the cumulative mass and store the total mass
        mass_shells= numpy.array([\
            integrate.quad(lambda r: _FOURPI*r**2*self.dens(r),
                           rlo,rhi)[0] for rlo,rhi in zip(r[:-1],r[1:])])
        self._cumul_mass= numpy.cumsum(mass_shells)
        self.mass= self._cumul_mass[-1]
        return None
        
    def _dens_W(self,W):
        """Density as a function of W"""
        sqW= numpy.sqrt(W)
        return numpy.exp(W)*special.erf(sqW)-_TWOOVERSQRTPI*sqW*(1.+2./3.*W)

    def dens(self,r):
        return self._dens_W(self._W_from_r(r))