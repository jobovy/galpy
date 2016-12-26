###############################################################################
#   DiskSCFPotential.py: Potential expansion for disk+halo potentials
###############################################################################
import numpy
from galpy.potential_src.Potential import Potential, _APY_LOADED
from galpy.potential_src.SCFPotential import SCFPotential, \
    scf_compute_coeffs_axi
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
                 dSigmadphi=None,
                 d2Sigmadphi2=None,
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
        # Parse given functions
        self.isNonAxi= dens.__code__.co_argcount == 3
        self._SigmaNonAxi= Sigma.__code__.co_argcount == 2
        # BOVY: CHECK THAT THESE ARE CONSISTENT (NO NONAXI SIGMA FOR AXI DENS)
        # Store approx. functions, currently assumes single profile, axi
        self._Sigma_amp= Sigma_amp
        self._Sigma= Sigma
        self._dSigmadR= dSigmadR
        self._d2SigmadR2= d2SigmadR2
        self._Hz= Hz
        self._hz= hz
        self._dHzdz= dHzdz
        self._inputdens= dens
        # Solve Poisson equation for Phi_ME
        if not self.isNonAxi:
            dens_func= lambda R,z: phiME_dens_axi(R,z,dens,
                                                  Sigma,dSigmadR,d2SigmadR2,
                                                  hz,Hz,dHzdz,Sigma_amp)
            Acos, Asin= scf_compute_coeffs_axi(dens_func,N,L,a=a)
            self._phiME_dens_func= dens_func
        self._scf= SCFPotential(amp=1.,Acos=Acos,Asin=Asin,a=a,ro=None,vo=None)
        return None

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
        return 4.*numpy.pi*self._Sigma_amp*self._Sigma(r)*self._Hz(z)\
            +self._scf(R,z,phi=phi,use_physical=False)

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
        return -4.*numpy.pi*self._Sigma_amp*self._dSigmadR(r)*self._Hz(z)*R/r\
            +self._scf.Rforce(R,z,phi=phi,use_physical=False)
        
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
        return -4.*numpy.pi*self._Sigma_amp\
            *(self._dSigmadR(r)*self._Hz(z)*z/r+self._Sigma(r)*self._dHzdz(z))\
            +self._scf.zforce(R,z,phi=phi,use_physical=False)
        
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
        r= numpy.sqrt(R**2.+z**2.)
        return -4.*numpy.pi*self._Sigma_amp\
            *self._dSigmadphi(r,phi)*self._Hz(z)\
            +self._scf.phiforce(R,z,phi=phi,use_physical=False)

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
        return 4.*numpy.pi*self._Sigma_amp*self._Hz(z)/r**2.\
            *(self._d2SigmadR2(r)*R**2.+z**2./r*self._dSigmadR(r))\
            +self._scf.R2deriv(R,z,phi=phi,use_physical=False)
        
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
        return 4.*numpy.pi*self._Sigma_amp*\
            (self._Hz(z)/r**2.*(self._d2SigmadR2(r)*z**2.
                                +self._dSigmadR(r)*R**2./r)
             +2.*self._dSigmadR(r)*self._dHzdz(z)*z/r
             +self._Sigma(r)*self._hz(z))\
             +self._scf.z2deriv(R,z,phi=phi,use_physical=False)       

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
        return 4.*numpy.pi*self._Sigma_amp*\
            (self._Hz(z)*R*z/r**2.*(self._d2SigmadR2(r)-self._dSigmadR(r)/r)
             +self._dSigmadR(r)*self._dHzdz(z)*R/r)\
            +self._scf.Rzderiv(R,z,phi=phi,use_physical=False)
        
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
        r= numpy.sqrt(R**2.+z**2.)
        return 4.*numpy.pi*self._Sigma_amp\
            *self._d2Sigmadphi2(r,phi)*self._Hz(z)\
            +self._scf.phi2deriv(R,z,phi=phi,use_physical=False)

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
        if self.isNonAxi:
            return self._inputdens(R,z,phi)
        else:
            return self._inputdens(R,z)

def phiME_dens_axi(R,z,dens,Sigma,dSigmadR,d2SigmadR2,hz,Hz,dHzdz,Sigma_amp):
    """The density corresponding to phi_ME in the axisymmetric case"""
    r= numpy.sqrt(R**2.+z**2.)
    return dens(R,z)\
        -Sigma_amp*(Sigma(r)*hz(z)+d2SigmadR2(r)*Hz(z)
                    +2./r*dSigmadR(r)*(Hz(z)+z*dHzdz(z)))
