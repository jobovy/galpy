
import numpy as nu
from galpy.potential_src.Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
    
from galpy.util import bovy_coords
from scipy.special import eval_gegenbauer, lpmn

class SCFPotential(Potential):
   
    def __init__(self, amp=1., Areal=nu.ones((10,10,10), float), Aimag=nu.ones((10,10,10), float) ,normalize=False, ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            initialize a SCF Potential

        INPUT:

            amp       - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass density or Gxmass density

            Areal - The real part of the expansion coefficent  (NxLxL matrix)
            
            Aimag - The imaginary part of the expansion coefficent (NxLxL matrix)
    
            normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           SCFPotential object

        HISTORY:

           2016-05-13 - Written - Aladdin 

        """        
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(Areal,units.Quantity): 
            Areal= Areal.to(units.kpc).value/self._ro 
        if _APY_LOADED and isinstance(Aimag,units.Quantity): 
            Aimag= Aimag.to(units.kpc).value/self._ro 
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): 
            self.normalize(normalize)
        self._Areal, self._Aimag = Areal, Aimag

        
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
        fact = nu.math.factorial
        NN = nu.zeros((L,L),float)
        for l in range(L):
            for m in range(l + 1):
                NN[l][m] = ((2*l + 1.)/(4*nu.pi) * fact(l - m)/fact(l + m))**.5 * (2. - (m==0))
        return NN
        
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
           2016-05-17 - Written - Aladdin 
        """
        xi = (1. - r)/(1. + r)
        rho = nu.zeros((N,L), float)
        for n in range(N):
            for l in range(L):
                K = 0.5 * n * (n + 4*l + 3) + (l + 1)*(2*l + 1)
                rho[n][l] = K/(2*nu.pi) * (r**l)/ (r*(1 + r)**(2*l + 3)) * eval_gegenbauer(n,2*l + 3./2, xi)* (4*nu.pi)**0.5
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
           2016-05-17 - Written - Aladdin 
        """
        xi = (1. - r)/(1. + r)
        phi = nu.zeros((N,L), float)
        for n in range(N):
            for l in range(L):
                phi[n][l] = - (r**l)/ ((1 + r)**(2*l + 1)) * eval_gegenbauer(n,2*l + 3./2, xi)* (4*nu.pi)**0.5
        return phi  
        
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
        Areal, Aimag = self._Areal, self._Aimag
        N, L, M = Areal.shape
        r, theta, phi = bovy_coords.cyl_to_spher(R,phi,z)
        NN = self._Nroot(L)

        PP = lpmn(L,L,nu.cos(theta))[0]
        rho = nu.zeros((N,L,L), float) ## rho_n,l,m
        rho_tilde = self._rhoTilde(r, N, L) ##tilde rho_n,l
        def coeff(l,m):
            for n in range(N):
                rh_tmp = rho_tilde[n][l]*(Areal[n][l][m]*nu.cos(m*phi) + Aimag[n][l][m]*nu.sin(m*phi))
                rho[n][l][m] = rh_tmp
        for l in range(L):
            for m in range(l + 1):
                coeff(l,m)
                rho[:,l,m]*= PP[l,m]
        return nu.sum(rho) 
       
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
        Areal, Aimag = self._Areal, self._Aimag
        r, theta, phi = bovy_coords.cyl_to_spher(R,phi,z)
        N, L, M = Areal.shape
        NN = self._Nroot(L)
        PP = lpmn(L,L,nu.cos(theta))
        Phi = nu.zeros((N,L,L), float) ## rho_n,l,m
        ##TODO
        Phi_tilde = self._phiTilde(r, N, L) ##tilde rho_n,l
        def coeff(l,m): ##Duplicate Code :(
            for n in range(N):
                Ph_tmp = Phi_tilde[n][l]*(Areal[n][l][m]*nu.cos(m*phi) + Aimag[n][l][m]*nu.sin(m*phi))
                
                Phi[n][l][m] = Ph_tmp
        for l in range(L):
            for m in range(l + 1):
                coeff(l,m)
                Phi[:,l,m]*= PP[l,m]
        return nu.sum(phi) 
