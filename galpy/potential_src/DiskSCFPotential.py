###############################################################################
#   DiskSCFPotential.py: Potential expansion for disk+halo potentials
###############################################################################
import numpy
from galpy.potential_src.Potential import Potential, _APY_LOADED
from galpy.potential_src.SCFPotential import SCFPotential, \
    scf_compute_coeffs_axi, scf_compute_coeffs
if _APY_LOADED:
    from astropy import units
class DiskSCFPotential(Potential):
    """Class that implements a basis-function-expansion technique for solving the Poisson equation for disk (+halo) systems"""
    def __init__(self,amp=1.,normalize=False,
                 dens= lambda R,z: 13.5*numpy.exp(-3.*R)\
                     *numpy.exp(-27.*numpy.fabs(z)),
                 Sigma_amp=1.,
                 Sigma= lambda R: numpy.exp(-3.*R),
                 dSigmadR= lambda R: -3.*numpy.exp(-3.*R),
                 d2SigmadR2= lambda R: 9.*numpy.exp(-3.*R),
                 hz= lambda z: 13.5*numpy.exp(-27.*numpy.fabs(z)),
                 Hz= lambda z: (numpy.exp(-27.*numpy.fabs(z))-1.
                                +27.*numpy.fabs(z))/54.,
                 dHzdz= lambda z: 0.5*numpy.sign(z)*\
                     (1.-numpy.exp(-27.*numpy.fabs(z))),
                 N=20,L=20,a=1.,
                 ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            initialize a diskSCF Potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           DiskSCFPotential object

        HISTORY:

           2016-12-26 - Written - Bovy (UofT)
        """        
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity): 
            a= a.to(units.kpc).value/self._ro
        # Parse and store given functions
        self.isNonAxi= dens.__code__.co_argcount == 3
        self._parse_Sigma(Sigma_amp,Sigma,dSigmadR,d2SigmadR2)
        self._parse_hz(hz,Hz,dHzdz)
        if self.isNonAxi:
            self._inputdens= dens
        else:
            self._inputdens= lambda R,z,phi: dens(R,z)
        # Solve Poisson equation for Phi_ME
        if not self.isNonAxi:
            dens_func= lambda R,z: phiME_dens(R,z,0.,self._inputdens,
                                              self._Sigma,self._dSigmadR,
                                              self._d2SigmadR2,
                                              self._hz,self._Hz,
                                              self._dHzdz,self._Sigma_amp)
            Acos, Asin= scf_compute_coeffs_axi(dens_func,N,L,a=a)
        else:
            dens_func= lambda R,z,phi: phiME_dens(R,z,phi,self._inputdens,
                                                  self._Sigma,self._dSigmadR,
                                                  self._d2SigmadR2,
                                                  self._hz,self._Hz,
                                                  self._dHzdz,self._Sigma_amp)
            Acos, Asin= scf_compute_coeffs(dens_func,N,L,a=a)
        self._phiME_dens_func= dens_func
        self._scf= SCFPotential(amp=1.,Acos=Acos,Asin=Asin,a=a,ro=None,vo=None)
        return None

    def _parse_Sigma(self,Sigma_amp,Sigma,dSigmadR,d2SigmadR2):
        """
        NAME:
           _parse_Sigma
        PURPOSE:
           Parse the various input options for Sigma* functions
        HISTORY:
           2016-12-27 - Written - Bovy (UofT/CCA)
        """
        try:
            nsigma= len(Sigma)
        except TypeError:
            Sigma_amp= [Sigma_amp]
            Sigma= [Sigma]
            dSigmadR= [dSigmadR]
            d2SigmadR2= [d2SigmadR2]
            nsigma= 1
        self._nsigma= nsigma
        if isinstance(Sigma[0],dict):
            pass
        self._Sigma_amp= Sigma_amp
        self._Sigma= Sigma
        self._dSigmadR= dSigmadR
        self._d2SigmadR2= d2SigmadR2
        return None
    
    def _parse_hz(self,hz,Hz,dHzdz):
        """
        NAME:
           _parse_hz
        PURPOSE:
           Parse the various input options for Sigma* functions
        HISTORY:
           2016-12-27 - Written - Bovy (UofT/CCA)
        """
        if isinstance(hz,dict):
            hz= [hz]
        try:
            nhz= len(hz)
        except TypeError:
            hz= [hz]
            Hz= [Hz]
            dHzdz= [dHzdz]
            nhz= 1
        if nhz != self._nsigma and nhz != 1:
            raise ValueError('Number of hz functions needs to be equal to the number of Sigma functions or to 1')
        if nhz == 1 and self._nsigma > 1:
            hz= [hz[0] for ii in range(self._nsigma)]
            Hz= [Hz[0] for ii in range(self._nsigma)]
            dHzdz= [dHzdz[0] for ii in range(self._nsigma)]
        self._Hz= Hz
        self._hz= hz
        self._dHzdz= dHzdz       
        self._nhz= len(self._hz)
        if isinstance(hz[0],dict):
            self._parse_hz_dict()
        return None

    def _parse_hz_dict(self):
        hz, Hz, dHzdz= [], [], []
        for ii in range(self._nhz):
            th, tH, tdH= self._parse_hz_dict_indiv(self._hz[ii])
            hz.append(th)
            Hz.append(tH)
            dHzdz.append(tdH)
        self._hz= hz
        self._Hz= Hz
        self._dHzdz= dHzdz
        return None

    def _parse_hz_dict_indiv(self,hz):
        htype= hz.get('type','exp')
        if htype == 'exp':
            zd= hz.get('h',0.0375)
            th= lambda z, tzd=zd: 1./2./tzd*numpy.exp(-numpy.fabs(z)/tzd)
            tH= lambda z, tzd= zd: (numpy.exp(-numpy.fabs(z)/tzd)-1.
                                    +numpy.fabs(z)/tzd)*tzd/2.
            tdH= lambda z, tzd= zd: 0.5*numpy.sign(z)\
                *(1.-numpy.exp(-numpy.fabs(z)/tzd))
        return (th,tH,tdH)
    
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
           2016-12-26 - Written - Bovy (UofT/CCA)
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        r= numpy.sqrt(R**2.+z**2.)
        out= self._scf(R,z,phi=phi,use_physical=False)
        for a,s,H in zip(self._Sigma_amp,self._Sigma,self._Hz):
            out+= 4.*numpy.pi*a*s(r)*H(z)
        return out

    def _Rforce(self,R,z,phi=0, t=0):
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
           2016-12-26 - Written - Bovy (UofT/CCA)
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        r= numpy.sqrt(R**2.+z**2.)
        out= self._scf.Rforce(R,z,phi=phi,use_physical=False)
        for a,ds,H in zip(self._Sigma_amp,self._dSigmadR,self._Hz):
            out-= 4.*numpy.pi*a*ds(r)*H(z)*R/r
        return out

    def _zforce(self,R,z,phi=0,t=0):
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
           2016-12-26 - Written - Bovy (UofT/CCA)
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        r= numpy.sqrt(R**2.+z**2.)
        out= self._scf.zforce(R,z,phi=phi,use_physical=False)
        for a,s,ds,H,dH in zip(self._Sigma_amp,self._Sigma,self._dSigmadR,
                             self._Hz,self._dHzdz):
            out-= 4.*numpy.pi*a*(ds(r)*H(z)*z/r+s(r)*dH(z))
        return out
        
    def _phiforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the azimuthal force
        HISTORY:
           2016-12-26 - Written - Bovy (UofT)
        """
        if not self.isNonAxi:
            phi= 0.
        return self._scf.phiforce(R,z,phi=phi,use_physical=False)

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2016-12-26 - Written - Bovy (UofT/CCA)
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        r= numpy.sqrt(R**2.+z**2.)
        out= self._scf.R2deriv(R,z,phi=phi,use_physical=False)
        for a,ds,d2s,H in zip(self._Sigma_amp,self._dSigmadR,self._d2SigmadR2,
                              self._Hz):
            out+= 4.*numpy.pi*a*H(z)/r**2.*(d2s(r)*R**2.+z**2./r*ds(r))
        return out
        
    def _z2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2016-12-26 - Written - Bovy (UofT/CCA)
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        r= numpy.sqrt(R**2.+z**2.)
        out= self._scf.z2deriv(R,z,phi=phi,use_physical=False)
        for a,s,ds,d2s,h,H,dH in zip(self._Sigma_amp,
                                   self._Sigma,self._dSigmadR,self._d2SigmadR2,
                                   self._hz,self._Hz,self._dHzdz):
            out+= 4.*numpy.pi*a*(H(z)/r**2.*(d2s(r)*z**2.+ds(r)*R**2./r)
                                 +2.*ds(r)*dH(z)*z/r+s(r)*h(z))
        return out

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2016-12-26 - Written - Bovy (UofT/CCA)
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        r= numpy.sqrt(R**2.+z**2.)
        out= self._scf.Rzderiv(R,z,phi=phi,use_physical=False)
        for a,ds,d2s,H,dH in zip(self._Sigma_amp,self._dsigmadR,
                                 self._d2SigmadR2,self._Hz,self._dHzdz):
            out+= 4.*numpy.pi*a*(H(z)*R*z/r**2.*(d2s(r)-ds(r)/r)
                                 +ds(r)*dH(z)*R/r)
        return out
        
    def _phi2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phi2deriv
        PURPOSE:
           evaluate the second azimuthal derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second azimuthal derivative
        HISTORY:
           2016-12-26 - Written - Bovy (UofT/CCA)
        """
        if not self.isNonAxi and phi is None:
            phi= 0.
        return self._scf.phi2deriv(R,z,phi=phi,use_physical=False)

    def _dens(self,R,z,phi=0.,t=0.):
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
           2016-12-26 - Written - Bovy (UofT/CCA)
        """
        return self._inputdens(R,z,phi)

def phiME_dens(R,z,phi,dens,Sigma,dSigmadR,d2SigmadR2,hz,Hz,dHzdz,Sigma_amp):
    """The density corresponding to phi_ME"""
    r= numpy.sqrt(R**2.+z**2.)
    out= dens(R,z,phi)
    for a,s,ds,d2s,h,H,dH \
            in zip(Sigma_amp,Sigma,dSigmadR,d2SigmadR2,hz,Hz,dHzdz):
        out-= a*(s(r)*h(z)+d2s(r)*H(z)+2./r*ds(r)*(H(z)+z*dH(z)))
    return out
