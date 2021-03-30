###############################################################################
#   AdiabaticContractionWrapperPotential.py: Wrapper to adiabatically
#                                            contract a DM halo in response
#                                            to the growth of a baryonic
#                                            component
###############################################################################
import numpy
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import fixed_point
from .Force import Force
from .interpSphericalPotential import interpSphericalPotential
from ..util import conversion
# Note: not actually implemented as a WrapperPotential!
class AdiabaticContractionWrapperPotential(interpSphericalPotential):
    """AdiabaticContractionWrapperPotential: Wrapper to adiabatically contract a DM halo in response to the growth of a baryonic component.
    """
    def __init__(self,amp=1.,pot=None,baryonpot=None,
                 method='cautun',f_bar=None,rmin=None,rmax=50.,
                 ro=None,vo=None):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a AdiabaticContractionWrapper Potential
        INPUT:
           amp - amplitude to be applied to the potential (default: 1.)
           pot - Potential instance or list thereof representing the density that is adiabatically contracted
           baryonpot - Potential instance or list thereof representing the density of baryons whose growth causes the contraction
           method= ('cautun') Type of adiabatic-contraction formula: 
                           'cautun' for that from Cautun et al. 2020 (2020MNRAS.494.4291C), 
                           'blumenthal' for that from Blumenthal et al. 1986 (1986ApJ...301...27B)
                           'gnedin' for that from Gnedin et al. 2004 (2004ApJ...616...16G)
           f_bar= (None) universal baryon fraction; if None, calculated from pot and baryonpot assuming 
                           that at rmax the halo contains the universal baryon fraction  
           rmin= (None) minimum radius to consider (default: rmax/2500)
           rmax= (50.) maximum radius to consider (can be Quantity)
           ro, vo= standard unit-conversion parameters
        OUTPUT:
           (none)
        HISTORY:
           2021-03-21 - Started based on Marius Cautun's code - Bovy (UofT)
        """
        # Initialize with Force just to parse (ro,vo)
        Force.__init__(self,ro=ro,vo=vo)
        rmax= conversion.parse_length(rmax,ro=self._ro)
        rmin= conversion.parse_length(rmin,ro=self._ro) if not rmin is None \
            else rmax/2500.
        # Compute baryon and DM enclosed masses on radial grid
        from ..potential import mass
        rgrid= numpy.concatenate((numpy.array([1.e-4]),
                                  numpy.geomspace(rmin,rmax,301)))
        baryon_mass= numpy.array([mass(baryonpot,r,use_physical=False)
                                  for r in rgrid])
        dm_mass= numpy.array([mass(pot,r,use_physical=False)
                              for r in rgrid])
        # Adiabatic contraction
        new_rforce = dm_mass/rgrid**2. 
        if f_bar is None:
            f_bar= baryon_mass[-1]/(baryon_mass[-1]+dm_mass[-1])
        if method.lower() == 'cautun':
            new_rforce *= _contraction_ratio_Cautun2020( dm_mass, baryon_mass, f_bar )
        elif method.lower() == 'gnedin':
            new_rforce *= _contraction_ratio_Gnedin2004(rgrid, dm_mass, baryon_mass, pot.rvir(overdens=180.,wrtcrit=False), f_bar )
        elif method.lower() == 'blumenthal':
            new_rforce *= _contraction_ratio_Blumenthal1986(rgrid, dm_mass, baryon_mass, f_bar )
#         new_rforce[0]= 0.
#         new_rforce_func= lambda r: -numpy.interp(r,rgrid,new_rforce)
        new_rforce_func= lambda r: -interp1d(rgrid,new_rforce,bounds_error=False,fill_value=(new_rforce[0],0.))(r)
        # Potential at zero = int_0^inf dr rforce
#         Phi0= integrate.quad(new_rforce_func,rgrid[0],numpy.inf)[0]
        Phi0= integrate.quad(new_rforce_func,rgrid[0],rgrid[-1],epsabs=0., epsrel=1.e-4,)[0] - new_rforce[-1]*rgrid[-1]
        interpSphericalPotential.__init__(self,
                                          rforce=new_rforce_func,
                                          rgrid=rgrid,
                                          Phi0=Phi0,
                                          ro=ro,vo=vo)
        

        
def _contraction_ratio_Cautun2020( M_DMO, M_bar, fbar ):
    # solve for the contracted enclosed DM mass
    func_M_DM_contract = lambda M, M_DMO, M_b: M_DMO *1.023*(M_DMO/(1.-fbar)/(M+M_b))**-0.540 
    M_DM = fixed_point( func_M_DM_contract, M_DMO.copy(), args=(M_DMO,M_bar), xtol=1.e-6 )
    return M_DM / M_DMO

def _contraction_ratio_Blumenthal1986( r, M_DMO, M_bar, fbar ):
    # solve for the coontracted radius 'rf' containing the same DM mass as enclosed for r 
    func_M_bar = interp1d( r, M_bar, bounds_error=False, fill_value=(M_bar[0],M_bar[-1]) )
#     func_M_bar = interp1d( r, M_bar, bounds_error=False, fill_value="extrapolate" )
    func_r_contract = lambda rf, ri, M_DM: ri * (M_DM/(1.-fbar)) / (M_DM+func_M_bar(rf)) 
    rf = fixed_point( func_r_contract, r.copy(), args=(r,M_DMO), xtol=1.e-6 )
    # now find how much the enclosed mass increased at r
    func_M_DM = interp1d( rf, M_DMO, bounds_error=False, fill_value=(M_DMO[0],M_DMO[-1]) )
    return func_M_DM(r) / M_DMO

def _contraction_ratio_Gnedin2004( r, M_DMO, M_bar, Rvir, fbar ):
    # solve for the coontracted radius 'rf' containing the same DM mass as enclosed for r 
    func_M_bar = interp1d( r, M_bar, bounds_error=False, fill_value=(M_bar[0],M_bar[-1]) )
    func_M_DMO = interp1d( r, M_DMO, bounds_error=False, fill_value=(M_DMO[0],M_DMO[-1]) )
    A, w = 0.85, 0.8
    func_r_mean = lambda ri: A*Rvir * (ri/Rvir)**w
    func_r_contract = lambda rf, ri, M_DM: ri * (M_DM/(1.-fbar)) / (M_DM+func_M_bar( func_r_mean(rf)) )
    M_DMO_rmean = func_M_DMO( func_r_mean(r) )
    rf = fixed_point( func_r_contract, r.copy(), args=(r,M_DMO_rmean), xtol=1.e-6 )
    # now find how much the enclosed mass increased at r
    func_M_DM = interp1d( rf, M_DMO, bounds_error=False, fill_value=(M_DMO[0],M_DMO[-1]) )
    return func_M_DM(r) / M_DMO
