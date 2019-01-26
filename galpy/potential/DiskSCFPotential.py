###############################################################################
#   DiskSCFPotential.py: Potential expansion for disk+halo potentials
###############################################################################
import copy
import numpy
import scipy
_SCIPY_VERSION= [int(v.split('rc')[0])
                 for v in scipy.__version__.split('.')]
if _SCIPY_VERSION[0] < 1 and _SCIPY_VERSION[1] < 10: #pragma: no cover
    from scipy.maxentropy import logsumexp
elif _SCIPY_VERSION[0] < 1 and _SCIPY_VERSION[1] < 19: #pragma: no cover
    from scipy.misc import logsumexp
else:
    from scipy.special import logsumexp
from .Potential import Potential, _APY_LOADED
from .SCFPotential import SCFPotential, \
    scf_compute_coeffs_axi, scf_compute_coeffs
if _APY_LOADED:
    from astropy import units
class DiskSCFPotential(Potential):
    """Class that implements a basis-function-expansion technique for solving the Poisson equation for disk (+halo) systems. We solve the Poisson equation for a given density :math:`\\rho(R,\phi,z)` by introducing *K* helper function pairs :math:`[\\Sigma_i(R),h_i(z)]`, with :math:`h_i(z) = \mathrm{d}^2 H(z) / \mathrm{d} z^2` and search for solutions of the form

    .. math::

       \Phi(R,\phi,z = \Phi_{\mathrm{ME}}(R,\phi,z) + 4\pi G\sum_i \\Sigma_i(r)\,H_i(z)\,,

where :math:`r` is the spherical radius :math:`r^2 = R^2+z^2`. We can solve for :math:`\Phi_{\mathrm{ME}}(R,\phi,z)` by solving

    .. math::

       \\frac{\\Delta \Phi_{\mathrm{ME}}(R,\phi,z)}{4\pi G} = \\rho(R,\phi,z) - \sum_i\left\{ \Sigma_i(r)\,h_i(z) + \\frac{\mathrm{d}^2 \Sigma_i(r)}{\mathrm{d} r^2}\,H_i(z)+\\frac{2}{r}\,\\frac{\mathrm{d} \Sigma_i(r)}{\mathrm{d} r}\left[H_i(z)+z\,\\frac{\mathrm{d}H_i(z)}{\mathrm{d} z}\\right]\\right\}\,.

We solve this equation by using the :ref:`SCFPotential <scf_potential>` class and methods (:ref:`scf_compute_coeffs_axi <scf_compute_coeffs_axi>` or :ref:`scf_compute_coeffs <scf_compute_coeffs>` depending on whether :math:`\\rho(R,\phi,z)` is axisymmetric or not). This technique works very well if the disk portion of the potential can be exactly written as :math:`\\rho_{\mathrm{disk}} = \sum_i \Sigma_i(R)\,h_i(z)`, because the effective density on the right-hand side of this new Poisson equation is then not 'disky' and can be well represented using spherical harmonics. But the technique is general and can be used to compute the potential of any disk+halo potential; the closer the disk is to :math:`\\rho_{\mathrm{disk}} \\approx \sum_i \Sigma_i(R)\,h_i(z)`, the better the technique works.

This technique was introduced by `Kuijken & Dubinski (1995) <http://adsabs.harvard.edu/abs/1995MNRAS.277.1341K>`__ and was popularized by `Dehnen & Binney (1998) <http://adsabs.harvard.edu/abs/1998MNRAS.294..429D>`__. The current implementation is a slight generalization of the technique in those papers and uses the SCF approach of `Hernquist & Ostriker (1992)
<http://adsabs.harvard.edu/abs/1992ApJ...386..375H>`__ to solve the Poisson equation for :math:`\Phi_{\mathrm{ME}}(R,\phi,z)` rather than solving it on a grid using spherical harmonics and interpolating the solution (as done in `Dehnen & Binney 1998 <http://adsabs.harvard.edu/abs/1998MNRAS.294..429D>`__).

    """
    def __init__(self,amp=1.,normalize=False,
                 dens= lambda R,z: 13.5*numpy.exp(-3.*R)\
                     *numpy.exp(-27.*numpy.fabs(z)),
                 Sigma={'type':'exp','h':1./3.,'amp':1.},
                 hz={'type':'exp','h':1./27.},
                 Sigma_amp=None,dSigmadR=None,d2SigmadR2=None,
                 Hz=None,dHzdz=None,
                 N=10,L=10,a=1.,radial_order=None,costheta_order=None,
                 phi_order=None,
                 ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            initialize a DiskSCF Potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); cannot have units currently

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

           dens= function of R,z[,phi optional] that gives the density [in natural units, cannot return a Quantity currently]

           N=, L=, a=, radial_order=, costheta_order=, phi_order= keywords setting parameters for SCF solution for Phi_ME (see :ref:`scf_compute_coeffs_axi <scf_compute_coeffs_axi>` or :ref:`scf_compute_coeffs <scf_compute_coeffs>` depending on whether :math:`\\rho(R,\phi,z)` is axisymmetric or not)

           Either:

              (a) Sigma= Dictionary of surface density (example: {'type':'exp','h':1./3.,'amp':1.,'Rhole':0.} for amp x exp(-Rhole/R-R/h) )

                  hz= Dictionary of vertical profile, either 'exp' or 'sech2' (example {'type':'exp','h':1./27.} for exp(-|z|/h)/[2h], sech2 is sech^2(z/[2h])/[4h])

              (b) Sigma= function of R that gives the surface density
              
                  dSigmadR= function that gives d Sigma / d R

                  d2SigmadR2= function that gives d^2 Sigma / d R^2

                  Sigma_amp= amplitude to apply to all Sigma functions

                  hz= function of z that gives the vertical profile

                  Hz= function of z such that d^2 Hz(z) / d z^2 = hz

                  dHzdz= function of z that gives d Hz(z) / d z

              In both of these cases lists of arguments can be given for multiple disk components; can't mix (a) and (b) in these lists;  if hz is a single item the same vertical profile is assumed for all Sigma

        OUTPUT:

           DiskSCFPotential object

        HISTORY:

           2016-12-26 - Written - Bovy (UofT)
        """        
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units=None)
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
            Acos, Asin= scf_compute_coeffs_axi(dens_func,N,L,a=a,
                                               radial_order=radial_order,
                                               costheta_order=costheta_order)
        else:
            dens_func= lambda R,z,phi: phiME_dens(R,z,phi,self._inputdens,
                                                  self._Sigma,self._dSigmadR,
                                                  self._d2SigmadR2,
                                                  self._hz,self._Hz,
                                                  self._dHzdz,self._Sigma_amp)
            Acos, Asin= scf_compute_coeffs(dens_func,N,L,a=a,
                                           radial_order=radial_order,
                                           costheta_order=costheta_order,
                                           phi_order=phi_order)
        self._phiME_dens_func= dens_func
        self._scf= SCFPotential(amp=1.,Acos=Acos,Asin=Asin,a=a,ro=None,vo=None)
        if not self._Sigma_dict is None and not self._hz_dict is None:
            self.hasC= True
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): 
            self.normalize(normalize)
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
        if isinstance(Sigma,dict):
            Sigma= [Sigma]
        try:
            nsigma= len(Sigma)
        except TypeError:
            Sigma_amp= [Sigma_amp]
            Sigma= [Sigma]
            dSigmadR= [dSigmadR]
            d2SigmadR2= [d2SigmadR2]
            nsigma= 1
        self._nsigma= nsigma
        self._Sigma_amp= Sigma_amp
        self._Sigma= Sigma
        self._dSigmadR= dSigmadR
        self._d2SigmadR2= d2SigmadR2
        if isinstance(Sigma[0],dict):
            self._Sigma_dict= copy.copy(Sigma)
            self._parse_Sigma_dict()
        else:
            self._Sigma_dict= None
        return None
    
    def _parse_Sigma_dict(self):
        Sigma_amp, Sigma, dSigmadR, d2SigmadR2= [], [], [], []
        for ii in range(self._nsigma):
            ta, ts, tds, td2s= self._parse_Sigma_dict_indiv(self._Sigma[ii])
            Sigma_amp.append(ta)
            Sigma.append(ts)
            dSigmadR.append(tds)
            d2SigmadR2.append(td2s)
        self._Sigma_amp= Sigma_amp
        self._Sigma= Sigma
        self._dSigmadR= dSigmadR
        self._d2SigmadR2= d2SigmadR2
        return None

    def _parse_Sigma_dict_indiv(self,Sigma):
        stype= Sigma.get('type','exp')
        if stype == 'exp' and not 'Rhole' in Sigma:
            rd= Sigma.get('h',1./3.)
            ta= Sigma.get('amp',1.)
            ts= lambda R, trd=rd: numpy.exp(-R/trd)
            tds= lambda R, trd=rd: -numpy.exp(-R/trd)/trd
            td2s= lambda R, trd=rd: numpy.exp(-R/trd)/trd**2.
        elif stype == 'expwhole' or (stype == 'exp' and 'Rhole' in Sigma):
            rd= Sigma.get('h',1./3.)
            rm= Sigma.get('Rhole',0.5)
            ta= Sigma.get('amp',1.)
            ts= lambda R, trd=rd, trm=rm: numpy.exp(-trm/R-R/trd)
            tds= lambda R, trd=rd, trm=rm: \
                (trm/R**2.-1./trd)*numpy.exp(-trm/R-R/trd)
            td2s= lambda R, trd=rd,trm=rm: \
                ((trm/R**2.-1./trd)**2.-2.*trm/R**3.)*numpy.exp(-trm/R-R/trd)
        return (ta,ts,tds,td2s)
    
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
            if not isinstance(hz[0],dict):
                Hz= [Hz[0] for ii in range(self._nsigma)]
                dHzdz= [dHzdz[0] for ii in range(self._nsigma)]
        self._Hz= Hz
        self._hz= hz
        self._dHzdz= dHzdz       
        self._nhz= len(self._hz)
        if isinstance(hz[0],dict):
            self._hz_dict= copy.copy(hz)
            self._parse_hz_dict()
        else:
            self._hz_dict= None
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
        elif htype == 'sech2':
            zd= hz.get('h',0.0375)
            th= lambda z, tzd=zd: 1./numpy.cosh(z/2./tzd)**2./4./tzd
            # Avoid overflow in cosh
            tH= lambda z, tzd= zd: \
                tzd*(logsumexp(numpy.array([z/2./tzd,-z/2./tzd]),axis=0)\
                         -numpy.log(2.))
            tdH= lambda z, tzd= zd: numpy.tanh(z/2./tzd)/2.
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
        return self._scf.phiforce(R,z,phi=phi,use_physical=False)

    def _R2deriv(self,R,z,phi=0.,t=0.): #pragma: no cover
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
        raise AttributeError
        # Implementation above does not work bc SCF.R2deriv is not implemented
        r= numpy.sqrt(R**2.+z**2.)
        out= self._scf.R2deriv(R,z,phi=phi,use_physical=False)
        for a,ds,d2s,H in zip(self._Sigma_amp,self._dSigmadR,self._d2SigmadR2,
                              self._Hz):
            out+= 4.*numpy.pi*a*H(z)/r**2.*(d2s(r)*R**2.+z**2./r*ds(r))
        return out
        
    def _z2deriv(self,R,z,phi=0.,t=0.): #pragma: no cover
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
        raise AttributeError
        # Implementation above does not work bc SCF.z2deriv is not implemented
        r= numpy.sqrt(R**2.+z**2.)
        out= self._scf.z2deriv(R,z,phi=phi,use_physical=False)
        for a,s,ds,d2s,h,H,dH in zip(self._Sigma_amp,
                                   self._Sigma,self._dSigmadR,self._d2SigmadR2,
                                   self._hz,self._Hz,self._dHzdz):
            out+= 4.*numpy.pi*a*(H(z)/r**2.*(d2s(r)*z**2.+ds(r)*R**2./r)
                                 +2.*ds(r)*dH(z)*z/r+s(r)*h(z))
        return out

    def _Rzderiv(self,R,z,phi=0.,t=0.): #pragma: no cover
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
        raise AttributeError
        # Implementation above does not work bc SCF.Rzderiv is not implemented
        r= numpy.sqrt(R**2.+z**2.)
        out= self._scf.Rzderiv(R,z,phi=phi,use_physical=False)
        for a,ds,d2s,H,dH in zip(self._Sigma_amp,self._dsigmadR,
                                 self._d2SigmadR2,self._Hz,self._dHzdz):
            out+= 4.*numpy.pi*a*(H(z)*R*z/r**2.*(d2s(r)-ds(r)/r)
                                 +ds(r)*dH(z)*R/r)
        return out
        
    def _phi2deriv(self,R,z,phi=0.,t=0.): #pragma: no cover
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
        raise AttributeError
        # Implementation above does not work bc SCF.phi2deriv is not implemented
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
        r= numpy.sqrt(R**2.+z**2.)
        out= self._scf.dens(R,z,phi=phi,use_physical=False)
        for a,s,ds,d2s,h,H,dH in zip(self._Sigma_amp,self._Sigma,
                                     self._dSigmadR,self._d2SigmadR2,
                                     self._hz,self._Hz,self._dHzdz):
            out+= a*(s(r)*h(z)+d2s(r)*H(z)+2./r*ds(r)*(H(z)+z*dH(z)))
        return out

def phiME_dens(R,z,phi,dens,Sigma,dSigmadR,d2SigmadR2,hz,Hz,dHzdz,Sigma_amp):
    """The density corresponding to phi_ME"""
    r= numpy.sqrt(R**2.+z**2.)
    out= dens(R,z,phi)
    for a,s,ds,d2s,h,H,dH \
            in zip(Sigma_amp,Sigma,dSigmadR,d2SigmadR2,hz,Hz,dHzdz):
        out-= a*(s(r)*h(z)+d2s(r)*H(z)+2./r*ds(r)*(H(z)+z*dH(z)))
    return out
