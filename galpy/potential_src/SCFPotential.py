
import numpy as nu
from galpy.potential_src.Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
    
from galpy.util import bovy_coords
from scipy.special import eval_gegenbauer, lpmn, gamma
from scipy.integrate import quad, nquad

import itertools

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

    def _C(self,xi, L, N):
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
        fact = nu.math.factorial
        CC = nu.zeros((N,L), float) 
        for l in range(L):
            for n in range(N):
                alpha = 2*l + 3./2.
                if n==0:
                    CC[n][l] = 1
                    continue 
                elif n==1: CC[n][l] = 2*alpha*xi
                if n + 1 != N:
                    CC[n+1][l] = (n + 1)**-1. * (2*(n + alpha)*xi*CC[n][l] - (n + 2*alpha - 1)*CC[n-1][l])
        return CC
 
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
        fact = nu.math.factorial
        NN = nu.zeros((L,L),float)
        for l in range(L):
            for m in range(l + 1):
                NN[l][m] = ((2*l + 1.)/(4*nu.pi) * fact(l - m)/fact(l + m))**.5 * (2. - (m==0))
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
        for n in range(N):
            for l in range(L):
                K = 0.5 * n * (n + 4*l + 3) + (l + 1)*(2*l + 1)
                rho[n][l] = K/(2*nu.pi) * (a*r)**l/ ((r/a)*(a + r)**(2*l + 3)) * CC[n][l]* (4*nu.pi)**0.5
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
        for n in range(N):
            for l in range(L):
                phi[n][l] = - (r*a)**l/ ((a + r)**(2*l + 1)) * CC[n,l]* (4*nu.pi)**0.5
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
        CC = self._C(xi,L,N)
        func_tilde = funcTilde(r, CC, N, L) ## Tilde of the function of interest 
        
        func = nu.zeros((N,L,L), float) ## The function of interest (density or potential)
        
        m = nu.arange(0, L)
        mcos = nu.cos(m*phi)
        msin = nu.sin(m*phi)
        for l in range(L):
            for m in range(l + 1):
                    func[:,l,m] = (func_tilde[:,l]*(Acos[:,l,m]*mcos[m] + Asin[:,l,m]*msin[m]))*PP[l,m]*NN[l,m]
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
            return a**3. * dens(R)*(1 + xi)**2. * (1 - xi)**-3. * eval_gegenbauer(n,3./2, xi)
               
        Acos = nu.zeros((N,1,1), float)
        Asin = nu.zeros((N,1,1), float)
        
        for n in range(N):
            K = 16*nu.pi*(n + 3./2)/((n + 2)*(n + 1)*(1 + n*(n + 3.)/2.))
            Acos[n,0,0] = K*quad(integrand, -1., 1.)[0]
        return Acos, Asin
        
def C(xi, L, N):
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
        fact = nu.math.factorial
        CC = nu.zeros((N,L), float) 
        for l in range(L):
            for n in range(N):
                alpha = 2*l + 3./2.
                if n==0:
                    CC[n][l] = 1
                    continue 
                elif n==1: CC[n][l] = 2*alpha*xi
                if n + 1 != N:
                    CC[n+1][l] = (n + 1)**-1. * (2*(n + alpha)*xi*CC[n][l] - (n + 2*alpha - 1)*CC[n-1][l])
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
        def integrand(xi, theta):
            l = nu.arange(0, L)
            l = nu.repeat([l], [N], axis=0)
            r = xiToR(xi)
            R, z, phi = bovy_coords.spher_to_cyl(r, theta, 0)
            Legandre = lpmn(L - 1,L - 1,nu.cos(theta))[0].T[:,0]
            Legandre = nu.repeat([Legandre], [N], axis=0)
            return dens(R,z)*(1 + xi)**(l + 2.) * nu.power((1 - xi),(l - 3.)) * C(xi, L, L)*nu.sin(theta)*Legandre
        
               
        Acos = nu.zeros((N,L,L), float)
        Asin = nu.zeros((N,L,L), float)
        
        ##This should save us some computation time since we're only taking the double integral once, rather then L times
        integrated = gaussianQuadrature(integrand, [[-1., 1.], [0, nu.pi]], shape=(N, L))
        
        for n in range(N):
            for l in range(L):
                K = .5*n*(n + 4*l + 3) + (l + 1)*(2*l + 1)
                I = -K*(4*nu.pi)/(2.)**(8*l + 6) * gamma(n + 4*l + 3)/(gamma(n + 1)*(n + 2*l + 3./2)*gamma(2*l + 3./2)**2)
                Acos[n,l,0] = -2**(-2*l) * I**-1. * 2*nu.pi*integrated[n,l] * (2*l + 1)**0.5
        return Acos, Asin
        
def gaussianQuadrature(integrand, bounds, N=100, shape=None):
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
           shape - the shape of the array that integrand returns. None if it returns a float
        OUTPUT:
           The integral of the function integrand 
        HISTORY:
           2016-05-24 - Written - Aladdin 
        """
    def gaussxw(N):
        a = nu.linspace(3,4*N-1,N)/(4*N+2)
        x = nu.cos(nu.pi*a+1/(8*N*N*nu.tan(a)))
        epsilon = 1e-15
        delta = 1.0
        while delta>epsilon:
            p0 = nu.ones(N,float)
            p1 = nu.copy(x)
            for k in range(1,N):
                p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
            dp = (N+1)*(p0-x*p1)/(1-x*x)
            dx = p1/dp
            x -= dx
            delta = max(abs(dx))
        # Calculate the weights
        w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)
        return x,w
        
    ##Calculates the sample points and weights
    x,w = gaussxw(N)
    
    ##Maps the sample point and weights
    xp = nu.zeros((len(bounds), N), float)
    wp = nu.zeros((len(bounds), N), float)
    for i in range(len(bounds)):
        bound = bounds[i]
        a,b = bound
        xp[i] = .5*(b-a)*x + .5*(b+a)
        wp[i] = .5*(b - a)*w

    ##Performs the actual integration
    
    s = 0
    if shape != None: ##This will allow us to integrate a matrix
        s = nu.zeros(shape, float)
    
    ## i is a tuple of size n(number of integrals we're taking), that iterates over the range of N
    for i in itertools.product(range(N), repeat=len(bounds)):
        index = [[],[]]
        for j in range(len(i)):
            index[0].append(j)
            index[1].append(i[j])
        s+= nu.prod(wp[index])*integrand(*xp[index])
   
    return s
                
        