# McMillan (2017) potential as first implemented in the galpy framework by
# Mackereth & Bovy (2018)
# 
# Implemented in this funny way here to allow lazy-loading
# (https://stackoverflow.com/questions/1462986/lazy-module-variables-can-it-be-done)
import sys
import os
import copy
def _setup_globals(): # this func necessary to transfer *all* globals in Py2
    out= copy.copy(globals())
    out['__path__']= [os.path.dirname(__file__)]
    return out
class _McMillan17(object):
    def __init__(self):
        self._pot= None
        # This is necessary to transfer *all* globals in Py2
        self.__globals__= _setup_globals()
        return None

    # For tab completion
    def __dir__(self):
        return ['McMillan17']

    @property
    def McMillan17(self):
        if not self._pot:
            self._pot= _setup_mcmillan17()
        return self._pot

    def __getattr__(self,name):
        try:
            #In Py3 you can just do 'return globals()[name]', but not in Py2
            return self.__globals__[name]
        except: raise AttributeError("'module' object has no attribute '{}'".format(name))

__all__= []

# The magic: makes it such that we can do from _mcmillan17 import McMillan17
sys.modules[__name__] = _McMillan17()

def _setup_mcmillan17():
    import numpy
    from galpy.potential import NFWPotential
    from galpy.potential import DiskSCFPotential
    from galpy.potential import SCFPotential
    from galpy.potential import scf_compute_coeffs_axi
    from galpy.potential import NumericalPotentialDerivativesMixin
    from galpy.util import bovy_conversion
    # Suppress the numpy floating-point warnings that this code generates...
    old_error_settings= numpy.seterr(all='ignore')
    # Unit normalizations
    ro= 8.21
    vo= 233.1
    sigo= bovy_conversion.surfdens_in_msolpc2(vo=vo,ro=ro)
    rhoo= bovy_conversion.dens_in_msolpc3(vo=vo,ro=ro)
    #gas disk parameters (fixed in McMillan 2017...)
    Rd_HI= 7./ro
    Rm_HI= 4./ro
    zd_HI= 0.085/ro
    Sigma0_HI= 53.1/sigo
    Rd_H2= 1.5/ro
    Rm_H2= 12./ro
    zd_H2= 0.045/ro
    Sigma0_H2= 2180./sigo
    #parameters of best-fitting model in McMillan (2017)
    #stellar disks
    Sigma0_thin= 896./sigo
    Rd_thin= 2.5/ro
    zd_thin= 0.3/ro
    Sigma0_thick= 183./sigo
    Rd_thick= 3.02/ro
    zd_thick= 0.9/ro
    #bulge
    rho0_bulge= 98.4/rhoo
    r0_bulge= 0.075/ro
    rcut= 2.1/ro
    #DM halo
    rho0_halo= 0.00854/rhoo
    rh= 19.6/ro

    def gas_dens(R,z):
        if R == 0.:
            return 0.
        HI_dens= Sigma0_HI/(4*zd_HI)\
            *numpy.exp(-Rm_HI/R-R/Rd_HI)*1./numpy.cosh(z/(2*zd_HI))**2
        H2_dens= Sigma0_H2/(4*zd_H2)\
            *numpy.exp(-Rm_H2/R-R/Rd_H2)*1./numpy.cosh(z/(2*zd_H2))**2
        return HI_dens+H2_dens

    def stellar_dens(R,z):
        thin_dens= Sigma0_thin/(2*zd_thin)\
            *numpy.exp(-numpy.fabs(z)/zd_thin-R/Rd_thin)
        thick_dens= Sigma0_thick/(2*zd_thick)\
            *numpy.exp(-numpy.fabs(z)/zd_thick-R/Rd_thick)
        return thin_dens+thick_dens

    def bulge_dens(R,z):
        rdash= numpy.sqrt(R**2+(z/0.5)**2)
        dens= rho0_bulge/(1+rdash/r0_bulge)**1.8*numpy.exp(-(rdash/rcut)**2)
        return dens

    #dicts used in DiskSCFPotential 
    sigmadict = [{'type':'exp','h':Rd_HI,'amp':Sigma0_HI, 'Rhole':Rm_HI},
                 {'type':'exp','h':Rd_H2,'amp':Sigma0_H2, 'Rhole':Rm_H2},
                 {'type':'exp','h':Rd_thin,'amp':Sigma0_thin, 'Rhole':0.},
                 {'type':'exp','h':Rd_thick,'amp':Sigma0_thick, 'Rhole':0.}]

    hzdict = [{'type':'sech2', 'h':zd_HI},
              {'type':'sech2', 'h':zd_H2},
              {'type':'exp', 'h':zd_thin},
              {'type':'exp', 'h':zd_thick}]

    # SCF and DiskSCF classes with numerical 2nd derivatives
    class SCFPotentialwNumericalSecondDerivatives(\
        SCFPotential,NumericalPotentialDerivativesMixin):
        def __init__(self,*args,**kwargs):
            NumericalPotentialDerivativesMixin.__init__(self,kwargs)
            SCFPotential.__init__(self,*args,**kwargs)
    class DiskSCFPotentialwNumericalSecondDerivatives(\
        DiskSCFPotential,NumericalPotentialDerivativesMixin):
        def __init__(self,*args,**kwargs):
            NumericalPotentialDerivativesMixin.__init__(self,kwargs)
            DiskSCFPotential.__init__(self,*args,**kwargs)

    #generate separate disk and halo potential - and combined potential
    McMillan_bulge= SCFPotentialwNumericalSecondDerivatives(\
        Acos=scf_compute_coeffs_axi(bulge_dens,20,10,a=0.1)[0],
        a=0.1,ro=ro,vo=vo)
    McMillan_disk= DiskSCFPotentialwNumericalSecondDerivatives(\
        dens=lambda R,z: gas_dens(R,z)+stellar_dens(R,z),
        Sigma=sigmadict,hz=hzdict,a=2.5,N=30,L=30,ro=ro,vo=vo)
    McMillan_halo= NFWPotential(amp=rho0_halo*(4*numpy.pi*rh**3),
                                 a=rh,ro=ro,vo=vo)
    # Go back to old floating-point warnings settings
    numpy.seterr(**old_error_settings)
    return McMillan_disk+McMillan_halo+McMillan_bulge
