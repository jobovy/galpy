
import numpy as nu
from galpy.potential_src.Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
    
from galpy.util import bovy_coords
from scipy.special import eval_gegenbauer, lpmn, gamma

from numpy.polynomial.legendre import leggauss

from scipy.special import gammaln


class SCFPotential(Potential):
   
    def __init__(self, amp=1., Acos=nu.ones((10,10,10), float), Asin=nu.ones((10,10,10), float), a = 1., normalize=False, ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            initialize a SCF Potential

        INPUT:

            amp       - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass density or Gxmass density

            Acos - The real part of the expansion coefficent  (NxLxL matrix)
            
            Asin - The imaginary part of the expansion coefficent (NxLxL matrix)
            
            a - scale length (can be Quantity)
    
            normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           SCFPotential object

        HISTORY:

           2016-05-13 - Written - Aladdin 

        """        
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='unitless')
        if _APY_LOADED and isinstance(a,units.Quantity): 
            a= a.to(units.kpc).value/self._ro 
            
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): 
            self.normalize(normalize)
        ##Acos and Asin must have the same shape
        self._Acos, self._Asin = Acos, Asin
        
        self._a = a

        self._NN = self._Nroot(Acos.shape[1]) ## We only ever need to compute this once
        return None

 
    def _Nroot(self, L):
        """
        NAME:
           _Nroot
        PURPOSE:
           Evaluate the square root of equation (3.15) with the (2 - del_m,0) term outside the square root
        INPUT:
           L - evaluate Nroot for 0 <= l <= L 
        OUTPUT:
           The square root of equation (3.15) with the (2 - del_m,0) outside
        HISTORY:
           2016-05-16 - Written - Aladdin 
        """
        NN = nu.zeros((L,L),float)
        l = nu.arange(0,L)[:,nu.newaxis]
        m = nu.arange(0,L)[nu.newaxis, :]
        nLn = gammaln(l-m+1) - gammaln(l+m+1)
        NN[:,:] = ((2*l+1.)/(4.*nu.pi) * nu.e**nLn)**.5 * 2
        NN[:,0] /= 2.
        NN = nu.tril(NN)
        return NN
    def _calculateXi(self, r):
        """
        NAME:
           _calculateXi
        PURPOSE:
           Calculate xi given r
        INPUT:
           r - Evaluate at radius r
        OUTPUT:
           xi
        HISTORY:
           2016-05-18 - Written - Aladdin 
        """
        a = self._a
        return  (r - a)/(r + a)  
    def _rhoTilde(self, r, CC, N,L):
        """
        NAME:
           _rhoTilde
        PURPOSE:
           Evaluate rho_tilde as defined in equation 3.9 and 2.24 for 0 <= n < N and 0 <= l < L
        INPUT:
           r - Evaluate at radius r
           CC - The Gegenbauer polynomial matrix
           N - size of the N dimension
           L - size of the L dimension
        OUTPUT:
           rho tilde 
        HISTORY:
           2016-05-17 - Written - Aladdin 
        """
        a = self._a
        rho = nu.zeros((N,L), float)
        n = nu.arange(0,N)[:, nu.newaxis]
        l = nu.arange(0, L)[nu.newaxis,:]
        K = 0.5 * n * (n + 4*l + 3) + (l + 1)*(2*l + 1)
        rho[:,:] = K/(2*nu.pi) * (a*r)**l/ ((r/a)*(a + r)**(2*l + 3)) * CC[:,:]* (4*nu.pi)**0.5
        return rho   

    def _phiTilde(self, r, CC, N,L):
        """
        NAME:
           _phiTilde
        PURPOSE:
           Evaluate phi_tilde as defined in equation 3.10 and 2.25 for 0 <= n < N and 0 <= l < L
        INPUT:
           r - Evaluate at radius r
           CC - The Gegenbauer polynomial matrix
           N - size of the N dimension
           L - size of the L dimension
        OUTPUT:
           phi tilde 
        HISTORY:
           2016-05-17 - Written - Aladdin 
        """
        a = self._a
        phi = nu.zeros((N,L), float)
        n = nu.arange(0,N)[:, nu.newaxis]
        l = nu.arange(0, L)[nu.newaxis,:]
        phi[:,:] = - (r*a)**l/ ((a + r)**(2*l + 1)) * CC[:,:]* (4*nu.pi)**0.5
        return phi  
        
    def _compute(self, funcTilde, R, z, phi):
        """
        NAME:
           _compute
        PURPOSE:
           evaluate the NxLxL density or potential
        INPUT:
           funcTidle - must be _rhoTilde or _phiTilde
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
        OUTPUT:
           An NxLxL density or potential at (R,z, phi)
        HISTORY:
           2016-05-18 - Written - Aladdin 
        """
        Acos, Asin = self._Acos, self._Asin
        N, L, M = Acos.shape    
        r, theta, phi = bovy_coords.cyl_to_spher(R,z,phi)
        xi = self._calculateXi(r)
        
        NN = self._NN
        PP = lpmn(L-1,L-1,nu.cos(theta))[0].T ##Get the Legendre polynomials
        CC = _C(xi,L,N)
        func_tilde = funcTilde(r, CC, N, L) ## Tilde of the function of interest 
        
        func = nu.zeros((N,L,L), float) ## The function of interest (density or potential)
        
        m = nu.arange(0, L)
        mcos = nu.cos(m*phi)
        msin = nu.sin(m*phi)
        func = func_tilde[:,:,None]*(Acos[:,:,:]*mcos[None,None, :] + Asin[:,:,:]*msin[None,None,:])*PP[None,:,:]*NN[None,:,:]
        return func
        
    def _dens(self, R, z, phi=0., t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density at (R,z, phi)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           density at (R,z, phi)
        HISTORY:
           2016-05-17 - Written - Aladdin 
        """
        return nu.sum(self._compute(self._rhoTilde, R, z, phi))
        
       
    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at (R,z, phi)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           potential at (R,z, phi)
        HISTORY:
           2016-05-17 - Written - Aladdin 
        """
        return nu.sum(self._compute(self._phiTilde, R, z, phi))



def xiToR(xi, a =1):
    return a*nu.divide((1. + xi),(1. - xi))    
        
def compute_coeffs_spherical(dens, N, a=1.):
        """
        NAME:
           _compute_coeffs_spherical
        PURPOSE:
           Numerically compute the expansion coefficients for a given spherical density
        INPUT:
           dens - A density function that takes a parameter R
           N - size of expansion coefficients
        OUTPUT:
           Expansion coefficients for density dens
        HISTORY:
           2016-05-18 - Written - Aladdin 
        """
        def integrand(xi):
            r = xiToR(xi, a)
            R = r
            
            return a**3. * dens(R)*(1 + xi)**2. * (1 - xi)**-3. * _C(xi, 1, N)[:,0]
               
        Acos = nu.zeros((N,1,1), float)
        Asin = nu.zeros((N,1,1), float)
        integrated = gaussianQuadrature(integrand, [[-1., 1.]])
        n = nu.arange(0,N)
        K = 16*nu.pi*(n + 3./2)/((n + 2)*(n + 1)*(1 + n*(n + 3.)/2.))
        Acos[n,0,0] = K*integrated
        return Acos, Asin
        
def _C(xi, L, N):
    """
    NAME:
       _C
    PURPOSE:
       Evaluate C_n,l (the Gegenbauer polynomial) for 0 <= l < L and 0<= n < N 
    INPUT:
       xi - radial transformed variable
       L - Size of the L dimension
       N - Size of the N dimension
    OUTPUT:
       An LxN Gegenbauer Polynomial 
    HISTORY:
       2016-05-16 - Written - Aladdin 
    """
    CC = nu.zeros((N,L), float) 
     
    for l in range(L):
        for n in range(N):
            alpha = 2*l + 3./2.
            if n==0:
                CC[n][l] = 1.
                continue 
            elif n==1: CC[n][l] = 2.*alpha*xi
            if n + 1 != N:
                CC[n+1][l] = (n + 1.)**-1. * (2*(n + alpha)*xi*CC[n][l] - (n + 2*alpha - 1)*CC[n-1][l])
    return CC       
        
def compute_coeffs_axi(dens, N, L):
        """
        NAME:
           _compute_coeffs_axi
        PURPOSE:
           Numerically compute the expansion coefficients for a given axi-symmetric density
        INPUT:
           dens - A density function that takes a parameter R and z
           N - size of the Nth dimension of the expansion coefficients
           L - size of the Lth dimension of the expansion coefficients
        OUTPUT:
           Expansion coefficients for density dens
        HISTORY:
           2016-05-20 - Written - Aladdin 
        """
        def integrand(xi, costheta):
            l = nu.arange(0, L)[nu.newaxis, :]
            r = xiToR(xi)
            theta = nu.arccos(costheta)
            R, z, phi = bovy_coords.spher_to_cyl(r, theta, 0)
            Legandre = lpmn(0,L-1,costheta)[0].T
            dV = 2.*(1. + xi)**2. * nu.power(1. - xi, -4.) 
            phi_nl = -2.**(-2*l - 1.)*(2*l + 1.)**.5 * (1. + xi)**l * (1. - xi)**(l + 1.)*_C(xi, L, N)[:,:] * Legandre[nu.newaxis,:,0]
            
            return dens(R,z) * phi_nl*dV
            
               
        Acos = nu.zeros((N,L,L), float)
        Asin = nu.zeros((N,L,L), float)
        
        ##This should save us some computation time since we're only taking the double integral once, rather then L times
        integrated = gaussianQuadrature(integrand, [[-1., 1.], [-1, 1]])*(2*nu.pi)
        n = nu.arange(0,N)[:,nu.newaxis]
        l = nu.arange(0,L)[nu.newaxis,:]
        K = .5*n*(n + 4*l + 3) + (l + 1)*(2*l + 1)
        #I = -K*(4*nu.pi)/(2.**(8*l + 6)) * gamma(n + 4*l + 3)/(gamma(n + 1)*(n + 2*l + 3./2)*gamma(2*l + 3./2)**2)
        ##Taking the ln of I will allow bigger size coefficients 
        lnI = -(8*l + 6)*nu.log(2) + gammaln(n + 4*l + 3) - gammaln(n + 1) - nu.log(n + 2*l + 3./2) - 2*gammaln(2*l + 3./2)
        I = -K*(4*nu.pi) * nu.e**(lnI)
        Acos[:,:,0] = I**-1 * integrated
        
        return Acos, Asin
        
def compute_coeffs(dens, N, L):
        """
        NAME:
           _compute_coeffs
        PURPOSE:
           Numerically compute the expansion coefficients for a given density
        INPUT:
           dens - A density function that takes a parameter R, z and phi
           N - size of the Nth dimension of the expansion coefficients
           L - size of the Lth and Mth dimension of the expansion coefficients
        OUTPUT:
           Expansion coefficients for density dens
        HISTORY:
           2016-05-27 - Written - Aladdin 
        """
        def integrand(xi, theta, phi):
            l = nu.arange(0, L)[nu.newaxis, :, nu.newaxis]
            m = nu.arange(0, L)[nu.newaxis,nu.newaxis,:]
            r = xiToR(xi)
            R, z, phi = bovy_coords.spher_to_cyl(r, theta, phi)
            Legandre = lpmn(L - 1,L-1,nu.cos(theta))[0].T[nu.newaxis,:,:]
            dV = 2.*(1. + xi)**2. * nu.power(1. - xi, -4.) * nu.sin(theta)
            
            Nln = .5*gammaln(l - m + 1) - .5*gammaln(l + m + 1) - (2*l + 1)*nu.log(2)
            NN = nu.e**(Nln)

            NN[nu.where(NN == nu.inf)] = 0 ## To account for the fact that m cant be bigger than l
            
            phi_nl = -NN* (2*l + 1.)**.5 * (1. + xi)**l * (1. - xi)**(l + 1.)*_C(xi, L, N)[:,:,nu.newaxis] * Legandre
            
            return dens(R,z, phi) * phi_nl[nu.newaxis, :,:,:]*nu.array([nu.cos(m*phi), nu.sin(m*phi)])*dV
            
               
        Acos = nu.zeros((N,L,L), float)
        Asin = nu.zeros((N,L,L), float)
        
        ##This should save us some computation time since we're only taking the Triple integral once, rather then L times
        integrated = gaussianQuadrature(integrand, [[-1., 1.], [0, nu.pi], [0, 2*nu.pi]])
        n = nu.arange(0,N)[:,nu.newaxis, nu.newaxis]
        l = nu.arange(0,L)[nu.newaxis,:, nu.newaxis]
        K = .5*n*(n + 4*l + 3) + (l + 1)*(2*l + 1)

        lnI = -(8*l + 6)*nu.log(2) + gammaln(n + 4*l + 3) - gammaln(n + 1) - nu.log(n + 2*l + 3./2) - 2*gammaln(2*l + 3./2)
        I = -K*(4*nu.pi) * nu.e**(lnI)
        Acos[:,:,:],Asin[:,:,:] = I[nu.newaxis,:,:,:]**-1 * integrated
        
        return Acos, Asin
        
def gaussianQuadrature(integrand, bounds, N=50, roundoff=0):
    """
        NAME:
           _gaussianQuadrature
        PURPOSE:
           Numerically take n integrals over a function that returns a float or an array 
        INPUT:
           integrand - The function you're integrating over.
           bounds - The bounds of the integral in the form of [[a_0, b_0], [a_1, b_1], ... , [a_n, b_n]] 
           where a_i is the lower bound and b_i is the upper bound
           N - Number of sample points
           roundoff - if the integral is less than this value, round it to 0.
        OUTPUT:
           The integral of the function integrand 
        HISTORY:
           2016-05-24 - Written - Aladdin 
    """
        
    ##Calculates the sample points and weights
    x,w = leggauss(N)
    
    ##Maps the sample point and weights
    xp = nu.zeros((len(bounds), N), float)
    wp = nu.zeros((len(bounds), N), float)
    for i in range(len(bounds)):
        bound = bounds[i]
        a,b = bound
        xp[i] = .5*(b-a)*x + .5*(b+a)
        wp[i] = .5*(b - a)*w

    
    ##Determines the shape of the integrand
    s = 0.
    shape=None
    s_temp = integrand(*nu.zeros(len(bounds)))
    if type(s_temp).__module__ == nu.__name__:
        shape = s_temp.shape
        s = nu.zeros(shape, float)
        
        
    ##Performs the actual integration
       

    def nuProduct(some_list, some_length):
        return nu.array(some_list)[nu.rollaxis(nu.indices((len(some_list),) * some_length), 0, some_length + 1)
        .reshape(-1, some_length)]
    
    li = nuProduct(nu.arange(0,N), len(bounds))
    for i in range(li.shape[0]):
        
        index = [nu.arange(len(bounds)),li[i]]
        s+= nu.prod(wp[index])*integrand(*xp[index])
    
    ##Rounds values that are less than roundoff to zero    
    if shape!= None:
        s[nu.where(nu.fabs(s) < roundoff)] = 0
    else: s *= nu.fabs(s) >roundoff
    return s
                
        