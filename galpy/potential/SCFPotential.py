import hashlib
import numpy
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.special import lpmn
from scipy.special import gammaln, gamma
from ..util import coords, conversion
from .Potential import Potential

from .NumericalPotentialDerivativesMixin import \
    NumericalPotentialDerivativesMixin

class SCFPotential(Potential,NumericalPotentialDerivativesMixin):
    """Class that implements the `Hernquist & Ostriker (1992) <http://adsabs.harvard.edu/abs/1992ApJ...386..375H>`_ Self-Consistent-Field-type potential. 
    Note that we divide the amplitude by 2 such that :math:`Acos = \\delta_{0n}\\delta_{0l}\\delta_{0m}` and :math:`Asin = 0` corresponds to :ref:`Galpy's Hernquist Potential <hernquist_potential>`.

    .. math::

        \\rho(r, \\theta, \\phi) = \\frac{amp}{2}\\sum_{n=0}^{\\infty} \\sum_{l=0}^{\\infty} \\sum_{m=0}^l N_{lm} P_{lm}(\\cos(\\theta))  \\tilde{\\rho}_{nl}(r) \\left(A_{cos, nlm} \\cos(m\\phi) + A_{sin, nlm} \\sin(m\\phi)\\right)
        
    where
    
    .. math::        
        
        \\tilde{\\rho}_{nl}(r) = \\frac{K_{nl}}{\\sqrt{\\pi}} \\frac{(a r)^l}{(r/a) (a + r)^{2l + 3}} C_{n}^{2l + 3/2}(\\xi)         
    .. math:: 
    
        \\Phi(r, \\theta, \\phi) = \\sum_{n=0}^{\\infty} \\sum_{l=0}^{\\infty} \\sum_{m=0}^l N_{lm} P_{lm}(\\cos(\\theta))  \\tilde{\\Phi}_{nl}(r) \\left(A_{cos, nlm} \\cos(m\\phi) + A_{sin, nlm} \\sin(m\\phi)\\right)
        
    where
        
    .. math::  
        \\tilde{\\Phi}_{nl}(r) = -\\sqrt{4 \\pi}K_{nl} \\frac{(ar)^l}{(a + r)^{2l + 1}} C_{n}^{2l + 3/2}(\\xi)     
        
        
    where
        
    .. math::  
    
        \\xi = \\frac{r - a}{r + a} \\qquad
        N_{lm} = \\sqrt{\\frac{2l + 1}{4\\pi} \\frac{(l - m)!}{(l + m)!}}(2 - \\delta_{m0}) \\qquad
        K_{nl} = \\frac{1}{2} n (n + 4l + 3) + (l + 1)(2l + 1)
    
    and :math:`P_{lm}` is the Associated Legendre Polynomials whereas :math:`C_n^{\\alpha}` is the Gegenbauer polynomial.
    """
    def __init__(self, amp=1., Acos=numpy.array([[[1]]]),Asin=None, a = 1., normalize=False, ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            initialize a SCF Potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           Acos - The real part of the expansion coefficent  (NxLxL matrix, or optionally NxLx1 if Asin=None)
            
           Asin - The imaginary part of the expansion coefficient (NxLxL matrix or None)
            
           a - scale length (can be Quantity)
    
           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           SCFPotential object

        HISTORY:

           2016-05-13 - Written - Aladdin Seaifan (UofT)

        """        
        NumericalPotentialDerivativesMixin.__init__(self,{}) # just use default dR etc.
        Potential.__init__(self,amp=amp/2.,ro=ro,vo=vo,amp_units='mass')
        a= conversion.parse_length(a,ro=self._ro)
        ##Errors
        shape = Acos.shape
        errorMessage = None
        if len(shape) != 3:
            errorMessage="Acos must be a 3 dimensional numpy array"
        elif Asin is not None and shape[1] != shape[2]:
            errorMessage="The second and third dimension of the expansion coefficients must have the same length"
        elif Asin is None and not (shape[2] == 1 or shape[1] == shape[2]):
            errorMessage="The third dimension must have length=1 or equal to the length of the second dimension"
        elif Asin is None and shape[1] > 1 and numpy.any(Acos[:,:,1:] !=0):
            errorMessage="Acos has non-zero elements at indices m>0, which implies a non-axi symmetric potential.\n" +\
            "Asin=None which implies an axi symmetric potential.\n" + \
            "Contradiction."
        elif Asin is not None and Asin.shape != shape:
            errorMessage = "The shape of Asin does not match the shape of Acos."
        if errorMessage is not None:
            raise RuntimeError(errorMessage)
            
        ##Warnings
        warningMessage=None
        if numpy.any(numpy.triu(Acos,1) != 0) or (Asin is not None and numpy.any(numpy.triu(Asin,1) != 0)):
            warningMessage="Found non-zero values at expansion coefficients where m > l\n" + \
            "The Mth and Lth dimension is expected to make a lower triangular matrix.\n" + \
            "All values found above the diagonal will be ignored."   
        if warningMessage is not None:  
            raise RuntimeWarning(warningMessage)            
        
        ##Is non axi?
        self.isNonAxi= True
        if Asin is None or shape[1] == 1 or (numpy.all(Acos[:,:,1:] == 0) and numpy.all(Asin[:,:,:]==0)):
            self.isNonAxi = False        
        
        
        self._a = a

        NN = self._Nroot(Acos.shape[1], Acos.shape[2])
        
        
        self._Acos= Acos*NN[numpy.newaxis,:,:]
        if Asin is not None:
            self._Asin = Asin*NN[numpy.newaxis,:,:]
        else:
            self._Asin = numpy.zeros_like(Acos)
        self._force_hash= None
        self.hasC= True
        self.hasC_dxdv=True
        self.hasC_dens=True
        
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): 
            self.normalize(normalize)
        return None

    def _Nroot(self, L, M=None):
        """
        NAME:
           _Nroot
        PURPOSE:
           Evaluate the square root of equation (3.15) with the (2 - del_m,0) term outside the square root
        INPUT:
           L - evaluate Nroot for 0 <= l <= L 
           M - evaluate Nroot for 0 <= m <= M 
        OUTPUT:
           The square root of equation (3.15) with the (2 - del_m,0) outside
        HISTORY:
           2016-05-16 - Written - Aladdin Seaifan (UofT)
        """
        if M is None: M =L
        NN = numpy.zeros((L,M),float)
        l = numpy.arange(0,L)[:,numpy.newaxis]
        m = numpy.arange(0,M)[numpy.newaxis, :]
        nLn = gammaln(l-m+1) - gammaln(l+m+1)
        NN[:,:] = ((2*l+1.)/(4.*numpy.pi) * numpy.e**nLn)**.5 * 2
        NN[:,0] /= 2.
        NN = numpy.tril(NN)
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
           2016-05-18 - Written - Aladdin Seaifan (UofT) 
        """
        a = self._a
        if r == 0:
            return -1
        else:
            return (1.-a/r)/(1.+a/r)
    
    def _rhoTilde(self, r, N,L):
        """
        NAME:
           _rhoTilde
        PURPOSE:
           Evaluate rho_tilde as defined in equation 3.9 and 2.24 for 0 <= n < N and 0 <= l < L
        INPUT:
           r - Evaluate at radius r
           N - size of the N dimension
           L - size of the L dimension
        OUTPUT:
           rho tilde 
        HISTORY:
           2016-05-17 - Written - Aladdin Seaifan (UofT) 
        """
        xi = self._calculateXi(r)
        CC = _C(xi,N,L)
        a = self._a
        rho = numpy.zeros((N,L), float)
        n = numpy.arange(0,N, dtype=float)[:, numpy.newaxis]
        l = numpy.arange(0, L, dtype=float)[numpy.newaxis,:]
        K = 0.5 * n * (n + 4*l + 3) + (l + 1.)*(2*l + 1)
        rho[:,:] = K * ((a*r)**l) / ((r/a)*(a + r)**(2*l + 3.)) * CC[:,:]* (numpy.pi)**-0.5
        return rho   

    def _phiTilde(self, r, N,L):
        """
        NAME:
           _phiTilde
        PURPOSE:
           Evaluate phi_tilde as defined in equation 3.10 and 2.25 for 0 <= n < N and 0 <= l < L
        INPUT:
           r - Evaluate at radius r
           N - size of the N dimension
           L - size of the L dimension
        OUTPUT:
           phi tilde 
        HISTORY:
           2016-05-17 - Written - Aladdin Seaifan (UofT) 
        """
        xi = self._calculateXi(r)
        CC = _C(xi,N,L)
        a = self._a
        phi = numpy.zeros((N,L), float)
        n = numpy.arange(0,N)[:, numpy.newaxis]
        l = numpy.arange(0, L)[numpy.newaxis,:]
        if r == 0:
            phi[:,:]= -1./a* CC[:,:]*(4*numpy.pi)**0.5
        else:
            phi[:,:] = - a**l*r**(-l-1.)/ ((1.+a/r)**(2*l + 1.)) * CC[:,:]* (4*numpy.pi)**0.5
        return phi  
        
    def _compute(self, funcTilde, R, z, phi):
        """
        NAME:
           _compute
        PURPOSE:
           evaluate the NxLxM density or potential
        INPUT:
           funcTidle - must be _rhoTilde or _phiTilde
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
        OUTPUT:
           An NxLxM density or potential at (R,z, phi)
        HISTORY:
           2016-05-18 - Written - Aladdin Seaifan (UofT) 
        """
        Acos, Asin = self._Acos, self._Asin
        N, L, M = Acos.shape    
        r, theta, phi = coords.cyl_to_spher(R,z,phi)
        
        
   
        PP = lpmn(M-1,L-1,numpy.cos(theta))[0].T ##Get the Legendre polynomials
        func_tilde = funcTilde(r, N, L) ## Tilde of the function of interest 
        
        func = numpy.zeros((N,L,M), float) ## The function of interest (density or potential)
        
        m = numpy.arange(0, M)[numpy.newaxis, numpy.newaxis, :]
        mcos = numpy.cos(m*phi)
        msin = numpy.sin(m*phi)
        func = func_tilde[:,:,None]*(Acos[:,:,:]*mcos + Asin[:,:,:]*msin)*PP[None,:,:]
        return func
        
        
    def _computeArray(self, funcTilde, R, z, phi):
        """
        NAME:
           _computeArray
        PURPOSE:
           evaluate the density or potential for a given array of coordinates
        INPUT:
           funcTidle - must be _rhoTilde or _phiTilde
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
        OUTPUT:
           density or potential evaluated at (R,z, phi)
        HISTORY:
           2016-06-02 - Written - Aladdin Seaifan (UofT) 
        """
        R = numpy.array(R,dtype=float); z = numpy.array(z,dtype=float); phi = numpy.array(phi,dtype=float);
        
        shape = (R*z*phi).shape
        if shape == (): return numpy.sum(self._compute(funcTilde, R,z,phi))
        R = R*numpy.ones(shape); z = z*numpy.ones(shape); phi = phi*numpy.ones(shape);
        func = numpy.zeros(shape, float)

        
        li = _cartesian(shape)
        for i in range(li.shape[0]):
            j = numpy.split(li[i], li.shape[1])
            func[j] = numpy.sum(self._compute(funcTilde, R[j][0],z[j][0],phi[j][0]))
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
           2016-05-17 - Written - Aladdin Seaifan (UofT) 
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        return self._computeArray(self._rhoTilde, R,z,phi)
              
    def _mass(self,R,z=None,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           evaluate the mass within R (and z) for this potential; if z=None, integrate spherical
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           t - time
        OUTPUT:
           the mass enclosed
        HISTORY:
           2021-03-09 - Written - Bovy (UofT)
           2021-03-18 - Switched to using Gauss' theorem - Bovy (UofT)
        """
        if not z is None: raise AttributeError # Hack to fall back to general
        # when integrating over spherical volume, all non-zero l,m vanish
        N= len(self._Acos)
        return R**2.*numpy.sum(self._Acos[:,0,0]*self._dphiTilde(R,N,1)[:,0])
        
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
           2016-05-17 - Written - Aladdin Seaifan (UofT) 
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        return self._computeArray(self._phiTilde, R,z,phi)

    def _dphiTilde(self, r, N, L):
        """
        NAME:
           _dphiTilde
        PURPOSE:
           Evaluate the derivative of phiTilde with respect to r
        INPUT:
           r - spherical radius
           N - size of the N dimension
           L - size of the L dimension
        OUTPUT:
           the derivative of phiTilde with respect to r
        HISTORY:
           2016-06-06 - Written - Aladdin Seaifan (UofT) 
        """
        a = self._a
        l = numpy.arange(0, L, dtype=float)[numpy.newaxis, :]
        n = numpy.arange(0, N, dtype=float)[:, numpy.newaxis]
        xi = self._calculateXi(r)
        dC = _dC(xi,N,L)
        return -(4*numpy.pi)**.5 * (numpy.power(a*r, l)*(l*(a + r)*numpy.power(r,-1) -(2*l + 1))/((a + r)**(2*l + 2))*_C(xi,N,L) + 
        a**-1*(1 - xi)**2 * (a*r)**l / (a + r)**(2*l + 1) *dC/2.)
        
        
    def _computeforce(self,R,z,phi=0,t=0):
        """
        NAME:
           _computeforce
        PURPOSE:
           Evaluate the first derivative of Phi with respect to R, z and phi
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           dPhi/dr, dPhi/dtheta, dPhi/dphi
        HISTORY:
           2016-06-07 - Written - Aladdin Seaifan (UofT) 
        """
        Acos, Asin = self._Acos, self._Asin
        N, L, M = Acos.shape    
        r, theta, phi = coords.cyl_to_spher(R,z,phi)
        new_hash= hashlib.md5(numpy.array([R, z,phi])).hexdigest()
        
        if new_hash == self._force_hash:
            dPhi_dr = self._cached_dPhi_dr  
            dPhi_dtheta = self._cached_dPhi_dtheta 
            dPhi_dphi = self._cached_dPhi_dphi
            
        else:        
            PP, dPP = lpmn(M-1,L-1,numpy.cos(theta)) ##Get the Legendre polynomials
            PP = PP.T[None,:,:]
            dPP = dPP.T[None,:,:]
            phi_tilde = self._phiTilde(r, N, L)[:,:,numpy.newaxis] 
            dphi_tilde = self._dphiTilde(r,N,L)[:,:,numpy.newaxis]
            
            m = numpy.arange(0, M)[numpy.newaxis, numpy.newaxis, :]
            mcos = numpy.cos(m*phi)
            msin = numpy.sin(m*phi)
            dPhi_dr = -numpy.sum((Acos*mcos + Asin*msin)*PP*dphi_tilde)
            dPhi_dtheta = -numpy.sum((Acos*mcos + Asin*msin)*phi_tilde*dPP*(-numpy.sin(theta)))
            dPhi_dphi =-numpy.sum(m*(Asin*mcos - Acos*msin)*phi_tilde*PP)
            
            self._force_hash = new_hash
            self._cached_dPhi_dr = dPhi_dr
            self._cached_dPhi_dtheta = dPhi_dtheta
            self._cached_dPhi_dphi = dPhi_dphi
        return dPhi_dr,dPhi_dtheta,dPhi_dphi
        
    def _computeforceArray(self,dr_dx, dtheta_dx, dphi_dx, R, z, phi):
        """
        NAME:
           _computeforceArray
        PURPOSE:
           evaluate the forces in the x direction for a given array of coordinates
        INPUT:
           dr_dx - the derivative of r with respect to the chosen variable x
           dtheta_dx - the derivative of theta with respect to the chosen variable x
           dphi_dx - the derivative of phi with respect to the chosen variable x
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           The forces in the x direction
        HISTORY:
           2016-06-02 - Written - Aladdin Seaifan (UofT) 
        """     
        R = numpy.array(R,dtype=float); z = numpy.array(z,dtype=float); phi = numpy.array(phi,dtype=float);
        shape = (R*z*phi).shape
        if shape == (): 
            dPhi_dr,dPhi_dtheta,dPhi_dphi = \
            self._computeforce(R,z,phi)
            return dr_dx*dPhi_dr + dtheta_dx*dPhi_dtheta +dPhi_dphi*dphi_dx
        
        R = R*numpy.ones(shape);
        z = z* numpy.ones(shape);
        phi = phi* numpy.ones(shape);
        force = numpy.zeros(shape, float)
        dr_dx = dr_dx*numpy.ones(shape); dtheta_dx = dtheta_dx*numpy.ones(shape);dphi_dx = dphi_dx*numpy.ones(shape);  
        li = _cartesian(shape)

        for i in range(li.shape[0]):
            j = tuple(numpy.split(li[i], li.shape[1]))
            dPhi_dr,dPhi_dtheta,dPhi_dphi = \
            self._computeforce(R[j][0],z[j][0],phi[j][0])
            force[j] = dr_dx[j][0]*dPhi_dr + dtheta_dx[j][0]*dPhi_dtheta +dPhi_dphi*dphi_dx[j][0]
        return force
    def _Rforce(self, R, z, phi=0, t=0):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force at (R,z, phi)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           radial force at (R,z, phi)
        HISTORY:
           2016-06-06 - Written - Aladdin Seaifan (UofT) 
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        r, theta, phi = coords.cyl_to_spher(R,z,phi)
        #x = R
        dr_dR = numpy.divide(R,r); dtheta_dR = numpy.divide(z,r**2); dphi_dR = 0
        return self._computeforceArray(dr_dR, dtheta_dR, dphi_dR, R,z,phi)
        
    def _zforce(self, R, z, phi=0., t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force at (R,z, phi)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           vertical force at (R,z, phi)
        HISTORY:
           2016-06-06 - Written - Aladdin Seaifan (UofT) 
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        r, theta, phi = coords.cyl_to_spher(R,z,phi)
        #x = z
        dr_dz = numpy.divide(z,r); dtheta_dz = numpy.divide(-R,r**2); dphi_dz = 0
        return self._computeforceArray(dr_dz, dtheta_dz, dphi_dz, R,z,phi)
        
    def _phiforce(self, R,z,phi=0,t=0):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuth force at (R,z, phi)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           azimuth force at (R,z, phi)
        HISTORY:
           2016-06-06 - Written - Aladdin Seaifan (UofT) 
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        r, theta, phi = coords.cyl_to_spher(R,z,phi)
        #x = phi
        dr_dphi = 0; dtheta_dphi = 0; dphi_dphi = 1
        return self._computeforceArray(dr_dphi, dtheta_dphi, dphi_dphi, R,z,phi)
        
    def OmegaP(self):
        return 0

        
def _xiToR(xi, a =1):
    return a*numpy.divide((1. + xi),(1. - xi))  

def _RToxi(r, a=1):
    out= numpy.divide((r/a-1.),(r/a+1.),where=True^numpy.isinf(r))
    if numpy.any(numpy.isinf(r)):
        if hasattr(r,'__len__'):
            out[numpy.isinf(r)]= 1.
        else:
            return 1.
    return out
        
def _C(xi,N,L,alpha=lambda x: 2*x + 3./2,singleL=False):
    """
    NAME:
       _C
    PURPOSE:
       Evaluate C_n,l (the Gegenbauer polynomial) for 0 <= l < L and 0<= n < N 
    INPUT:
       xi - radial transformed variable
       N - Size of the N dimension
       L - Size of the L dimension
       alpha = A lambda function of l. Default alpha = 2l + 3/2 
       singleL= (False), if True only compute the L-th polynomial
    OUTPUT:
       An LxN Gegenbauer Polynomial 
    HISTORY:
       2016-05-16 - Written - Aladdin Seaifan (UofT) 
       2021-02-22 - Upgraded to array xi - Bovy (UofT)
       2021-02-22 - Added singleL for use in compute...nbody - Bovy (UofT)
    """
    floatIn= False
    if isinstance(xi,(float,int)):
        floatIn= True
        xi= numpy.array([xi])
    if singleL:
        Ls= [L]
    else:
        Ls= range(L)
    CC= numpy.zeros((N,len(Ls),len(xi)))
    for l,ll in enumerate(Ls):
        for n in range(N):
            a= alpha(ll)
            if n==0:
                CC[n,l]= 1.
                continue 
            elif n==1:
                CC[n,l]= 2.*a*xi
            if n + 1 != N:
                CC[n+1,l]= (2*(n + a)*xi*CC[n,l]-(n + 2*a - 1)*CC[n-1,l])\
                    /(n+1.)
    if floatIn:
        return CC[:,:,0]
    else:
        return CC 
    
def _dC(xi, N, L):
    l = numpy.arange(0,L)[numpy.newaxis, :]
    CC = _C(xi,N + 1,L, alpha = lambda x: 2*x + 5./2)
    CC = numpy.roll(CC, 1, axis=0)[:-1,:]
    CC[0, :] = 0
    CC *= 2*(2*l + 3./2)
    return CC

def scf_compute_coeffs_spherical_nbody(pos,N,mass=1.,a=1.):
    """        
    NAME:

       scf_compute_coeffs_spherical_nbody

    PURPOSE:

       Numerically compute the expansion coefficients for a spherical expansion for a given $N$-body set of points 

    INPUT:

       pos - positions of particles in rectangular coordinates with shape [3,n]
           
       N - size of the Nth dimension of the expansion coefficients

       mass= (1.) mass of particles (scalar or array with size n)

       a= (1.) parameter used to scale the radius

    OUTPUT:

       (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

    HISTORY:

       2020-11-18 - Written - Morgan Bennett (UofT)

       2021-02-22 - Sped-up - Bovy (UofT)

    """
    Acos = numpy.zeros((N,1,1), float)
    Asin = None
    r= numpy.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)
    RhoSum= numpy.einsum('j,ij',mass/(1.+r/a),_C(_RToxi(r,a=a),N,1)[:,0])
    n = numpy.arange(0,N)
    K = 4*(n + 3./2)/((n + 2)*(n + 1)*(1 + n*(n + 3.)/2.))
    Acos[n,0,0] = 2*K*RhoSum   
    return Acos, Asin 

def scf_compute_coeffs_spherical(dens, N, a=1., radial_order=None):
    """
    NAME:

       scf_compute_coeffs_spherical

    PURPOSE:

       Numerically compute the expansion coefficients for a given spherical density

    INPUT:

       dens - A density function that takes a parameter R

       N - size of expansion coefficients
           
       a= (1.) parameter used to scale the radius

       radial_order - Number of sample points of the radial integral. If None, radial_order=max(20, N + 1)

    OUTPUT:

       (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

    HISTORY:

       2016-05-18 - Written - Aladdin Seaifan (UofT) 

    """
    numOfParam = 0
    try:
        dens(0)
        numOfParam=1
    except:
        try:
            dens(0,0)
            numOfParam=2
        except:
            numOfParam=3
    param = [0]*numOfParam;
    
    def integrand(xi):
        r = _xiToR(xi, a)
        R = r
        param[0] = R
        return a**3. * dens(*param)*(1 + xi)**2. * (1 - xi)**-3. * _C(xi, N, 1)[:,0]
               
    Acos = numpy.zeros((N,1,1), float)
    Asin = None
    
    Ksample = [max(N + 1, 20)]
    
    if radial_order != None:
        Ksample[0] = radial_order
            
    integrated = _gaussianQuadrature(integrand, [[-1., 1.]], Ksample=Ksample)
    n = numpy.arange(0,N)
    K = 16*numpy.pi*(n + 3./2)/((n + 2)*(n + 1)*(1 + n*(n + 3.)/2.))
    Acos[n,0,0] = 2*K*integrated
    return Acos, Asin    
        
def scf_compute_coeffs_axi_nbody(pos,N,L,mass=1.,a=1.):
    """        
    NAME:

       scf_compute_coeffs_axi_nbody

    PURPOSE:

       Numerically compute the expansion coefficients for a given $N$-body set of points assuming that the density is axisymmetric

    INPUT:

       pos - positions of particles in rectangular coordinates with shape [3,n]
           
       N - size of the Nth dimension of the expansion coefficients

       L - size of the Lth dimension of the expansion coefficients

       mass= (1.) mass of particles (scalar or array with size n)

       a= (1.) parameter used to scale the radius

    OUTPUT:

       (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

    HISTORY:

       2021-02-22 - Written based on general code - Bovy (UofT)

    """
    r= numpy.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)
    costheta = pos[2]/r
    mass= numpy.atleast_1d(mass)
    Acos, Asin= numpy.zeros([N,L,1]), None
    Pll= numpy.ones(len(r)) # Set up Assoc. Legendre recursion
    # (n,l) dependent constant
    n= numpy.arange(0,N)[:,numpy.newaxis]
    l= numpy.arange(0,L)[numpy.newaxis,:]
    Knl= 0.5*n*(n+4.*l+3.)+(l+1)*(2.*l+1.)
    Inl= -Knl*2.*numpy.pi/2.**(8.*l+6.)*gamma(n+4.*l+3.)\
        /gamma(n+1)/(n+2.*l+1.5)/gamma(2.*l+1.5)**2/numpy.sqrt(2.*l+1)
    # Set up Assoc. Legendre recursion
    Plm= Pll
    Plmm1= 0.
    for ll in range(L):
        # Compute Gegenbauer polys for this l
        Cn= _C(_RToxi(r,a=a),N,ll,singleL=True)
        phinlm= -(r/a)**ll/(1.+r/a)**(2.*ll+1)*Cn[:,0]*Plm
        # Acos
        Sum= numpy.sum(mass[numpy.newaxis,:]*phinlm,axis=-1)
        Acos[:,ll,0]= Sum/Inl[:,ll]
        # Recurse Assoc. Legendre
        if ll < L:
            tmp= Plm
            Plm= ((2*ll+1.)*costheta*Plm-ll*Plmm1)/(ll+1)
            Plmm1= tmp
    return Acos,Asin

def scf_compute_coeffs_axi(dens, N, L, a=1.,radial_order=None, costheta_order=None):
    """
    NAME:

       scf_compute_coeffs_axi

    PURPOSE:

       Numerically compute the expansion coefficients for a given axi-symmetric density

    INPUT:

       dens - A density function that takes a parameter R and z

       N - size of the Nth dimension of the expansion coefficients

       L - size of the Lth dimension of the expansion coefficients

       a - parameter used to shift the basis functions

       radial_order - Number of sample points of the radial integral. If None, radial_order=max(20, N + 3/2L + 1)

       costheta_order - Number of sample points of the costheta integral. If None, If costheta_order=max(20, L + 1)

    OUTPUT:

       (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

    HISTORY:

       2016-05-20 - Written - Aladdin Seaifan (UofT) 

    """
    numOfParam = 0
    try:
        dens(0,0)
        numOfParam=2
    except:
        numOfParam=3
    param = [0]*numOfParam;
    def integrand(xi, costheta):
        l = numpy.arange(0, L)[numpy.newaxis, :]
        r = _xiToR(xi,a)
        R = r*numpy.sqrt(1 - costheta**2.)
        z = r*costheta
        Legandre = lpmn(0,L-1,costheta)[0].T[numpy.newaxis,:,0]
        dV = (1. + xi)**2. * numpy.power(1. - xi, -4.) 
        phi_nl =  a**3*(1. + xi)**l * (1. - xi)**(l + 1.)*_C(xi, N, L)[:,:] * Legandre
        param[0] = R
        param[1] = z
        return  phi_nl*dV * dens(*param)

    Acos = numpy.zeros((N,L,1), float)
    Asin = None
        
    ##This should save us some computation time since we're only taking the double integral once, rather then L times
    Ksample = [max(N + 3*L//2 + 1, 20) ,  max(L + 1,20) ]
    if radial_order != None:
        Ksample[0] = radial_order
    if costheta_order != None:
        Ksample[1] = costheta_order
            
        
    integrated = _gaussianQuadrature(integrand, [[-1, 1], [-1, 1]], Ksample = Ksample)*(2*numpy.pi)
    n = numpy.arange(0,N)[:,numpy.newaxis]
    l = numpy.arange(0,L)[numpy.newaxis,:]
    K = .5*n*(n + 4*l + 3) + (l + 1)*(2*l + 1)
    #I = -K*(4*numpy.pi)/(2.**(8*l + 6)) * gamma(n + 4*l + 3)/(gamma(n + 1)*(n + 2*l + 3./2)*gamma(2*l + 3./2)**2)
    ##Taking the ln of I will allow bigger size coefficients 
    lnI = -(8*l + 6)*numpy.log(2) + gammaln(n + 4*l + 3) - gammaln(n + 1) - numpy.log(n + 2*l + 3./2) - 2*gammaln(2*l + 3./2)
    I = -K*(4*numpy.pi) * numpy.e**(lnI)
    constants = -2.**(-2*l)*(2*l + 1.)**.5 
    Acos[:,:,0] = 2*I**-1 * integrated*constants
    return Acos, Asin
    
def scf_compute_coeffs_nbody(pos,N,L,mass=1.,a=1.):
    """        
    NAME:

       scf_compute_coeffs_nbody

    PURPOSE:

       Numerically compute the expansion coefficients for a given $N$-body set of points

    INPUT:

       pos - positions of particles in rectangular coordinates with shape [3,n]
           
       N - size of the Nth dimension of the expansion coefficients

       L - size of the Lth and Mth dimension of the expansion coefficients

       mass= (1.) mass of particles (scalar or array with size n)

       a= (1.) parameter used to scale the radius

    OUTPUT:

       (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

    HISTORY:

       2020-11-18 - Written - Morgan Bennett (UofT)

    """ 
    r= numpy.sqrt(pos[0]**2+pos[1]**2+pos[2]**2)
    phi= numpy.arctan2(pos[1],pos[0])
    costheta= pos[2]/r
    sintheta= numpy.sqrt(1.-costheta**2.)
    mass= numpy.atleast_1d(mass)
    Acos, Asin= numpy.zeros([N,L,L]), numpy.zeros([N,L,L])
    Pll= numpy.ones(len(r)) # Set up Assoc. Legendre recursion
    # (n,l) dependent constant
    n= numpy.arange(0,N)[:,numpy.newaxis]
    l= numpy.arange(0,L)[numpy.newaxis,:]
    Knl= 0.5*n*(n+4.*l+3.)+(l+1)*(2.*l+1.)
    Inl= -Knl*2.*numpy.pi/2.**(8.*l+6.)*gamma(n+4.*l+3.)\
        /gamma(n+1)/(n+2.*l+1.5)/gamma(2.*l+1.5)**2
    for mm in range(L): # Loop over m
        cosmphi= numpy.cos(phi*mm)
        sinmphi= numpy.sin(phi*mm)
        # Set up Assoc. Legendre recursion
        Plm= Pll
        Plmm1= 0.
        for ll in range(mm,L):
            # Compute Gegenbauer polys for this l
            Cn= _C(_RToxi(r,a=a),N,ll,singleL=True)
            phinlm= -(r/a)**ll/(1.+r/a)**(2.*ll+1)*Cn[:,0]*Plm
            # Acos
            Sum= numpy.sqrt((2.*ll+1)*gamma(ll-mm+1)/gamma(ll+mm+1))\
                *numpy.sum((mass*cosmphi)[numpy.newaxis,:]*phinlm,axis=-1)
            Acos[:,ll,mm]= Sum/Inl[:,ll]
            # Asin
            Sum= numpy.sqrt((2.*ll+1)*gamma(ll-mm+1)/gamma(ll+mm+1))\
                *numpy.sum((mass*sinmphi)[numpy.newaxis,:]*phinlm,axis=-1)
            Asin[:,ll,mm]= Sum/Inl[:,ll]
            # Recurse Assoc. Legendre
            if ll < L:
                tmp= Plm
                Plm= ((2*ll+1.)*costheta*Plm-(ll+mm)*Plmm1)/(ll-mm+1)
                Plmm1= tmp
        # Recurse Assoc. Legendre
        Pll*= -(2*mm+1.)*sintheta        
    return Acos,Asin
        
def scf_compute_coeffs(dens,N,L,a=1.,
                       radial_order=None,costheta_order=None,phi_order=None):
    """        
    NAME:

       scf_compute_coeffs

    PURPOSE:

       Numerically compute the expansion coefficients for a given triaxial density

    INPUT:

       dens - A density function that takes a parameter R, z and phi

       N - size of the Nth dimension of the expansion coefficients

       L - size of the Lth and Mth dimension of the expansion coefficients
           
       a - parameter used to shift the basis functions

       radial_order - Number of sample points of the radial integral. If None, radial_order=max(20, N + 3/2L + 1)

       costheta_order - Number of sample points of the costheta integral. If None, If costheta_order=max(20, L + 1)

       phi_order - Number of sample points of the phi integral. If None, If costheta_order=max(20, L + 1)

    OUTPUT:

       (Acos,Asin) - Expansion coefficients for density dens that can be given to SCFPotential.__init__

    HISTORY:

       2016-05-27 - Written - Aladdin Seaifan (UofT)

    """
    def integrand(xi, costheta, phi):
        l = numpy.arange(0, L)[numpy.newaxis, :, numpy.newaxis]
        m = numpy.arange(0, L)[numpy.newaxis,numpy.newaxis,:]
        r = _xiToR(xi, a)
        R = r*numpy.sqrt(1 - costheta**2.)
        z = r*costheta
        Legandre = lpmn(L - 1,L-1,costheta)[0].T[numpy.newaxis,:,:]
        dV = (1. + xi)**2. * numpy.power(1. - xi, -4.)
        
        phi_nl = - a**3*(1. + xi)**l * (1. - xi)**(l + 1.)*_C(xi, N, L)[:,:,numpy.newaxis] * Legandre
            
        return dens(R,z, phi) * phi_nl[numpy.newaxis, :,:,:]*numpy.array([numpy.cos(m*phi), numpy.sin(m*phi)])*dV
            
    Acos = numpy.zeros((N,L,L), float)
    Asin = numpy.zeros((N,L,L), float)
           
    Ksample = [max(N + 3*L//2 + 1,20), max(L + 1,20 ), max(L + 1,20)]
    if radial_order != None:
        Ksample[0] = radial_order
    if costheta_order != None:
        Ksample[1] = costheta_order
    if phi_order != None:
        Ksample[2] = phi_order
    integrated = _gaussianQuadrature(integrand, [[-1., 1.], [-1., 1.], [0, 2*numpy.pi]], Ksample = Ksample)
    n = numpy.arange(0,N)[:,numpy.newaxis, numpy.newaxis]
    l = numpy.arange(0,L)[numpy.newaxis,:, numpy.newaxis]
    m = numpy.arange(0,L)[numpy.newaxis,numpy.newaxis,:]
    K = .5*n*(n + 4*l + 3) + (l + 1)*(2*l + 1)
        
    Nln = .5*gammaln(l - m + 1) - .5*gammaln(l + m + 1) - (2*l)*numpy.log(2)
    NN = numpy.e**(Nln)

    NN[numpy.where(NN == numpy.inf)] = 0 ## To account for the fact that m cant be bigger than l
            
    constants = NN*(2*l + 1.)**.5
        
    lnI = -(8*l + 6)*numpy.log(2) + gammaln(n + 4*l + 3) - gammaln(n + 1) - numpy.log(n + 2*l + 3./2) - 2*gammaln(2*l + 3./2)
    I = -K*(4*numpy.pi) * numpy.e**(lnI)
    Acos[:,:,:],Asin[:,:,:] = 2*(I**-1.)[numpy.newaxis,:,:,:] * integrated * constants[numpy.newaxis,:,:,:]
        
    return Acos, Asin

def _cartesian(arraySizes, out=None):
    """
    NAME:
        cartesian
    PURPOSE:
        Generate a cartesian product of input arrays.
    INPUT: 
        arraySizes - list of size of arrays
        out - Array to place the cartesian product in.
    OUTPUT: 
        2-D array of shape (product(arraySizes), len(arraySizes)) containing cartesian products
        formed of input arrays.
    HISTORY:
        2016-06-02 - Obtained from
        http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """
    arrays = []
    for i in range(len(arraySizes)):
        arrays.append(numpy.arange(0, arraySizes[i]))

    arrays = [numpy.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
   
    n = numpy.prod([x.size for x in arrays])
    if out is None:
        out = numpy.zeros([n, len(arrays)], dtype=dtype)
    
    m = n // arrays[0].size
    out[:,0] = numpy.repeat(arrays[0], m)
    if arrays[1:]:
        _cartesian(arraySizes[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
                
def _gaussianQuadrature(integrand, bounds, Ksample=[20], roundoff=0):
    """
        NAME:
           _gaussianQuadrature
        PURPOSE:
           Numerically take n integrals over a function that returns a float or an array 
        INPUT:
           integrand - The function you're integrating over.
           bounds - The bounds of the integral in the form of [[a_0, b_0], [a_1, b_1], ... , [a_n, b_n]] 
           where a_i is the lower bound and b_i is the upper bound
           Ksample - Number of sample points in the form of [K_0, K_1, ..., K_n] where K_i is the sample point
           of the ith integral.
           roundoff - if the integral is less than this value, round it to 0.
        OUTPUT:
           The integral of the function integrand 
        HISTORY:
           2016-05-24 - Written - Aladdin Seaifan (UofT) 
    """
     
    ##Maps the sample point and weights
    xp = numpy.zeros((len(bounds), numpy.max(Ksample)), float)
    wp = numpy.zeros((len(bounds), numpy.max(Ksample)), float)
    for i in range(len(bounds)):
        x,w = leggauss(Ksample[i]) ##Calculates the sample points and weights
        a,b = bounds[i]
        xp[i, :Ksample[i]] = .5*(b-a)*x + .5*(b+a)
        wp[i, :Ksample[i]] = .5*(b - a)*w

    
    ##Determines the shape of the integrand
    s = 0.
    shape=None
    s_temp = integrand(*numpy.zeros(len(bounds)))
    if type(s_temp).__name__ == numpy.ndarray.__name__ :
        shape = s_temp.shape
        s = numpy.zeros(shape, float)
        
        
    

    #gets all combinations of indices from each integrand 
    li = _cartesian(Ksample)
    
    ##Performs the actual integration
    for i in range(li.shape[0]):
        index = (numpy.arange(len(bounds)),li[i])
        s+= numpy.prod(wp[index])*integrand(*xp[index])
    
    ##Rounds values that are less than roundoff to zero    
    if shape!= None:
        s[numpy.where(numpy.fabs(s) < roundoff)] = 0
    else: s *= numpy.fabs(s) >roundoff
    return s
                
        
