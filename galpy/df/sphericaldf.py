# Superclass for spherical distribution functions, contains
#   - sphericaldf: superclass of all spherical DFs
#   - anisotropicsphericaldf: superclass of all anisotropic spherical DFs
import numpy
import pdb
import scipy.interpolate
from .df import df, _APY_LOADED
from ..potential import flatten as flatten_potential
from ..potential import evaluatePotentials, vesc
from ..orbit import Orbit
from ..util.bovy_conversion import physical_conversion
if _APY_LOADED:
    from astropy import units

class sphericaldf(df):
    """Superclass for spherical distribution functions"""
    def __init__(self,pot=None,scale=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initializes a spherical DF

        INPUT:

            pot - Spherical potential which determines the DF

            scale - Characteristic scale radius to aid sampling calculations. 
                Not necessary, and will also be overridden by value from pot if 
                available.

        OUTPUT:

            None

        HISTORY:

            2020-07-22 - Written - 

        """
        df.__init__(self,ro=ro,vo=vo)
        if pot is None:
            raise IOError("pot= must be set")
        # Some sort of check for spherical symmetry in the potential?
        assert not isinstance(pot,(list,tuple)), 'Lists of potentials not yet supported'
        self._pot = pot
        self._potInf = evaluatePotentials(pot,10**12,0)
        try:
            self._scale = pot._scale
        except AttributeError:
            if scale is not None:
                if _APY_LOADED and isinstance(scale,units.Quantity):
                    scale= scale.to(u.kpc).value/self._ro
                self._scale = scale
            else:
                self._scale = 1.
        self._xi_cmf_interpolator = self._make_cmf_interpolator()
        self._v_vesc_pvr_interpolator = self._make_pvr_interpolator()

############################## EVALUATING THE DF###############################
    @physical_conversion('phasespacedensity',pop=True)
    def __call__(self,*args,**kwargs):
        """
        NAME:

            __call__

        PURPOSE:

            return the DF

        INPUT:

            Either:

                a) (E,L,Lz): tuple of E and (optionally) L and (optionally) Lz.
                    Each may be Quantity
                    
                b) R,vR,vT,z,vz,phi: 

                c) Orbit instance: orbit.Orbit instance and if specific time 
                    then orbit.Orbit(t) 

        KWARGS:

            return_fE= if True then return the full distribution function plus 
                just the energy component (for e.g. an anisotropic DF)

        OUTPUT:

            Value of DF

        HISTORY:

            2020-07-22 - Written -

        """
        # Stub for calling the DF as a function of either a) R,vR,vT,z,vz,phi,
        # b) Orbit, c) E,L (Lz?) --> maybe depends on the actual form?
        
        # Get E,L,Lz. Generic requirements for any possible spherical DF?
        if len(args) == 1:
            if not isinstance(args[0],Orbit): # Assume tuple (E,L,Lz)
                if len(args[0]) == 1:
                    E = args[0][0]
                    L,Lz = None,None
                elif len(args[0]) == 2:
                    E,L = args[0]
                    Lz = None
                elif len(args[0]) == 3:
                    E,L,Lz = args[0]
            else: # Orbit
                E = args[0].E(pot=self._pot)
                L = numpy.sqrt(numpy.sum(numpy.square(args[0].L())))
                Lz = args[0].Lz()
            if _APY_LOADED and isinstance(E,units.Quantity):
                E= E.to(units.km**2/units.s**2).value/self._vo**2.
            if _APY_LOADED and isinstance(L,units.Quantity):
                L= L.to(units.kpc*units.km/units.s).value/self._ro/self._vo
            if _APY_LOADED and isinstance(Lz,units.Quantity):
                Lz= Lz.to(units.kpc*units.km/units.s).value/self._ro/self._vo
        else: # Assume R,vR,vT,z,vz,(phi)
            if len(args) == 5:
                R,vR,vT,z,vz = args
                phi = None
            else:
                R,vR,vT,z,vz,phi = args
            if _APY_LOADED and isinstance(R,units.Quantity):
                R= R.to(units.kpc).value/self._ro
            if _APY_LOADED and isinstance(vR,units.Quantity):
                vR= vR.to(units.km/units.s).value/self._vo
            if _APY_LOADED and isinstance(vT,units.Quantity):
                vT= vT.to(units.km/units.s).value/self._vo
            if _APY_LOADED and isinstance(z,units.Quantity):
                z= z.to(units.kpc).value/self._ro
            if _APY_LOADED and isinstance(vz,units.Quantity):
                vz= vz.to(units.km/units.s).value/self._vo
            if _APY_LOADED and isinstance(phi,units.Quantity):
                phi= phi.to(units.rad).value
            vtotSq = vR**2.+vT**2.+vz**2.
            E = 0.5*vtotSq + evaluatePotentials(R,z)
            Lz = R*vT
            r = numpy.sqrt(R**2.+z**2.)
            vrad = (R*vR+z*vz)/r
            L = numpy.sqrt(vtotSq-vrad**2.)*r
        f = self._call_internal(E,L,Lz) # Some function for each sub-class
        return f

############################### SAMPLING THE DF################################
    def sample(self,R=None,z=None,phi=None,n=1,return_orbit=True):
        """
        NAME:

            sample

        PURPOSE:

            Return full 6D samples of the DF

        INPUT:

            R= Radius at which to generate samples (can be Quantity)

            z= Height at which to generate samples (can be Quantity)
            
            phi= Azimuth at which to generate samples (can be Quantity)

            n= number of samples to generate

        OPTIONAL INPUT:

            return_orbit= If True output is orbit.Orbit object, if False 
                output is (R,vR,vT,z,vz,phi)

        OUTPUT:

            List of samples. Either vector (R,vR,vT,z,vz,phi) or orbit.Orbit

        NOTES:

            If R,z,phi are None then sample positions with CMF. If R,z,phi are 
            floats then sample n velocities at location. If array then sample 
            velocities at radii, ignoring n. phi can be None if R,z are set 
            by any above mechanism, will then sample phi for output.

        HISTORY:

            2020-07-22 - Written - 

        """
        if R is None or z is None: # Full 6D samples
            r = self._sample_r(n=n)
            v = self._sample_v(r,n=n)
            phi,theta = self._sample_position_angles(n=n)
            R = r*numpy.sin(theta)
            z = r*numpy.cos(theta) 
        else: # 3D velocity samples
            if isinstance(R,numpy.ndarray):
                assert len(R) == len(z)
                n = len(R)
            r = numpy.sqrt(R**2.+z**2.)
            v = self._sample_v(r,n=n)
            if phi is None: # Otherwise assume phi input type matches R,z
                phi,_ = self._sample_position_angles(n=n)
        
        eta,psi = self._sample_velocity_angles(n=n)
        vr = v*numpy.cos(eta)
        vtheta = v*numpy.sin(eta)*numpy.cos(psi)
        vT = v*numpy.sin(eta)*numpy.sin(psi)
        vR = vr*numpy.sin(theta) + vtheta*numpy.cos(theta)
        vz = vr*numpy.cos(theta) - vtheta*numpy.sin(theta)
        
        if return_orbit:
            o = Orbit(vxvv=numpy.array([R,vR,vT,z,vz,phi]).T,
                      ro=self._ro,vo=self._vo)
            return o
        else:
            return (R,vR,vT,z,vz,phi)

    def _sample_r(self,n=1):
        """Generate radial position samples from potential
        Note - the function interpolates the normalized CMF onto the variable 
        xi defined as:
        
        .. math:: \\xi = \\frac{r-1}{r+1}
        
        so that xi is in the range [-1,1], which corresponds to an r range of 
        [0,infinity)"""
        rand_mass_frac = numpy.random.random(size=n)
        if '_icmf' in dir(self):
            r_samples = self._icmf(rand_mass_frac)
        else:
            xi_samples = self._xi_cmf_interpolator(rand_mass_frac)
            r_samples = self._xi_to_r(xi_samples,a=self._scale)
        return r_samples

    def _make_cmf_interpolator(self):
        """Create the interpolator object for calculating radii from the CMF
        Note - must use self.xi_to_r() on any output of interpolator
        Note - the function interpolates the normalized CMF onto the variable 
        xi defined as:
        
        .. math:: \\xi = \\frac{r-1}{r+1}
        
        so that xi is in the range [-1,1], which corresponds to an r range of 
        [0,infinity)"""
        xis = numpy.arange(-1,1,1e-6)
        rs = self._xi_to_r(xis,a=self._scale)
        ms = self._pot.mass(rs,use_physical=False)
        ms /= self._pot.mass(10**12,use_physical=False)
        xis = numpy.append(xis,1)
        ms = numpy.append(ms,1)
        xis_cmf_interp = scipy.interpolate.interp1d(ms,xis,
            kind='cubic',bounds_error=False,fill_value='extrapolate')
        return xis_cmf_interp

    def _xi_to_r(self,xi,a=1):
        """Calculate r from xi"""
        return a*numpy.divide(1+xi,1-xi)
    
    def r_to_xi(self,r,a=1):
        """Calculate xi from r"""
        return numpy.divide(r/a-1,r/a+1)

    def _sample_position_angles(self,n=1):
        """Generate spherical angle samples"""
        phi_samples = numpy.random.uniform(size=n)*2*numpy.pi
        theta_samples = numpy.arccos(2*numpy.random.uniform(size=n)-1)
        return phi_samples,theta_samples

    def _sample_v(self,r,n=1):
        """Generate velocity samples"""
        vesc_vals = vesc(self._pot,r,use_physical=False)
        pvr_icdf_samples = numpy.random.random(size=n)
        v_vesc_samples = self._v_vesc_pvr_interpolator(numpy.log10(r/self._scale),
            pvr_icdf_samples,grid=False)
        return v_vesc_samples*vesc_vals

    def _sample_velocity_angles(self,n=1):
        """Generate samples of angles that set radial vs tangential velocities"""
        eta_samples = self._sample_eta(n)
        psi_samples = numpy.random.uniform(size=n)*2*numpy.pi
        return eta_samples,psi_samples

    def _make_pvr_interpolator(self, r_a_start=-3, r_a_end=3, 
        r_a_interval=0.05, v_vesc_interval=0.01, set_interpolator=True,
        output_grid=False):
        '''
        NAME:

        _make_pvr_interpolator

        PURPOSE:

        Calculate a grid of the velocity sampling function v^2*f(E) over many 
        radii. The radii are fractional with respect to some scale radius 
        which characteristically describes the size of the potential, 
        and the velocities are fractional with respect to the escape velocity 
        at each radius r. This information is saved in a 2D interpolator which 
        represents the inverse cumulative distribution at many radii. This 
        allows for sampling of v/vesc given an input r/a

        INPUT:

            r_a_start= radius grid start location in units of r/a

            r_a_end= radius grid end location in units of r/a

            r_a_interval= radius grid spacing in units of r/a

            v_vesc_interval= velocity grid spacing in units of v/vesc

        OUTPUT:

            None (But sets self._v_vesc_pvr_interpolator)

        HISTORY:

            Written 2020-07-24 - James Lane (UofT)
        '''
        # Make an array of r/a by v/vesc and then orbits to calculate fE
        r_a_values = numpy.power(10,numpy.arange(r_a_start,r_a_end,r_a_interval))
        v_vesc_values = numpy.arange(0,1,v_vesc_interval)
        r_a_grid, v_vesc_grid = numpy.meshgrid(r_a_values,v_vesc_values)
        vesc_grid = vesc(self._pot,r_a_grid*self._scale,use_physical=False)
        E_grid = evaluatePotentials(self._pot,r_a_grid*self._scale,0,
            use_physical=False)+0.5*(numpy.multiply(v_vesc_grid,vesc_grid))**2.

        # Calculate cumulative p(v|r)
        fE_grid = self.fE(E_grid).reshape(E_grid.shape)
        _beta = 0
        if hasattr(self,'beta'):
            _beta = self.beta
        pvr_grid = numpy.multiply(fE_grid,(v_vesc_grid*vesc_grid)**(2-2*_beta))
        pvr_grid_cml = numpy.cumsum( pvr_grid, axis=0 )
        pvr_grid_cml_norm = pvr_grid_cml\
        /numpy.repeat(pvr_grid_cml[-1,:][:,numpy.newaxis],pvr_grid_cml.shape[0],axis=1).T
        
        # Construct the inverse cumulative distribution
        n_new_pvr = 100 # Must be multiple of r_a_grid.shape[0]
        icdf_pvr_grid_reg = numpy.zeros((n_new_pvr,len(r_a_values)))
        icdf_v_vesc_grid_reg = numpy.zeros((n_new_pvr,len(r_a_values)))
        r_a_grid_reg = numpy.repeat(r_a_grid,n_new_pvr/r_a_grid.shape[0],axis=0)
        for i in range(pvr_grid_cml_norm.shape[1]):
            cml_pvr = pvr_grid_cml_norm[:,i]
            cml_pvr_inv_interp = scipy.interpolate.interp1d(cml_pvr, 
                v_vesc_values, kind='cubic', bounds_error=None, 
                fill_value='extrapolate')
            pvr_samples_reg = numpy.linspace(0,1,num=n_new_pvr)
            v_vesc_samples_reg = cml_pvr_inv_interp(pvr_samples_reg)
            icdf_pvr_grid_reg[:,i] = pvr_samples_reg
            icdf_v_vesc_grid_reg[:,i] = v_vesc_samples_reg
        ###i
        
        # Create the interpolator
        v_vesc_icdf_interpolator = scipy.interpolate.RectBivariateSpline(
            numpy.log10(r_a_grid[0,:]), icdf_pvr_grid_reg[:,0],
            icdf_v_vesc_grid_reg.T)
        return v_vesc_icdf_interpolator

class anisotropicsphericaldf(sphericaldf):
    """Superclass for anisotropic spherical distribution functions"""
    def __init__(self,pot=None,dftype=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize an anisotropic distribution function

        INPUT:

            dftype= Type of anisotropic DF, either 'constant' for constant beta 
                over all r, or 'ossipkov-merrit'

            pot - Spherical potential which determines the DF

        OUTPUT:
        
            None

        HISTORY:

            2020-07-22 - Written - 

        """
        self._dftype = dftype
        sphericaldf.__init__(self,pot=pot,ro=ro,vo=vo)