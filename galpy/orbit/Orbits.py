import os
import copy
import warnings
import numpy
from scipy import interpolate
from .Orbit import Orbit, _check_integrate_dt, _check_potential_dim, \
    _check_consistent_units
from .OrbitTop import _check_roSet, _check_voSet, _helioXYZ, _lbd, _radec, \
    _XYZvxvyvz, _lbdvrpmllpmbb, _pmrapmdec
from ..util import galpyWarning, galpyWarningVerbose
from ..util.bovy_conversion import physical_conversion
from ..util.bovy_coords import _K
from ..util.multi import parallel_map
from ..util.bovy_plot import _add_ticks
from ..util import bovy_conversion
from ..potential import toPlanarPotential, PotentialError, evaluatePotentials,\
    evaluateplanarPotentials, evaluatelinearPotentials
from ..potential import flatten as flatten_potential
from ..potential.Potential import _check_c
from ..potential.DissipativeForce import _isDissipative
from .integrateLinearOrbit import integrateLinearOrbit_c, _ext_loaded
from .integratePlanarOrbit import integratePlanarOrbit_c
from .integrateFullOrbit import integrateFullOrbit_c
from .. import actionAngle
ext_loaded= _ext_loaded
try:
    from astropy.coordinates import SkyCoord
    _APY_LOADED = True
except ImportError:
    SkyCoord = None
    _APY_LOADED = False
if _APY_LOADED:
    from astropy import units, coordinates
    import astropy
    _APY3= astropy.__version__ > '3'
from galpy.util import config
_APY_UNITS= config.__config__.getboolean('astropy','astropy-units')
# Set default numcores for integrate w/ parallel map using OMP_NUM_THREADS
try:
    _NUMCORES= int(os.environ['OMP_NUM_THREADS'])
except KeyError:
    import multiprocessing
    _NUMCORES= multiprocessing.cpu_count()
class Orbits(object):
    """
    Class representing multiple orbits.
    """
    def __init__(self, vxvv=[None],ro=None,vo=None,zo=None,solarmotion=None,
                 radec=False,uvw=False,lb=False):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize an Orbits instance

        INPUT:

            vxvv - initial conditions (must all have the same phase-space dimension); can be either

                a) list of Orbit instances

                b) astropy (>v3.0) SkyCoord including velocities (note that this turns *on* physical output even if ro and vo are not given)

                c) list of initial conditions for individual Orbit instances; elements can be either

                    1) in Galactocentric cylindrical coordinates [R,vR,vT(,z,vz,phi)]; can be Quantities

                    2) None: assumed to be the Sun (equivalent to ``[0,0,0,0,0,0]`` and ``radec=True``)


        OPTIONAL INPUTS:

            ro - distance from vantage point to GC (kpc; can be Quantity)

            vo - circular velocity at ro (km/s; can be Quantity)

            zo - offset toward the NGP of the Sun wrt the plane (kpc; can be Quantity; default = 25 pc)

            solarmotion - 'hogg' or 'dehnen', or 'schoenrich', or value in [-U,V,W]; can be Quantity

        OUTPUT:

            instance

        HISTORY:

            2018-10-13 - Written - Mathew Bub (UofT)

            2018-01-01 - Better handling of unit/coordinate-conversion parameters and consistency checks - Bovy (UofT)

            2018-02-01 - Handle array of SkyCoords in a faster way by making use of the fact that array of SkyCoords is processed correctly by Orbit

            2018-02-18 - Don't support radec, lb, or uvw keywords to avoid slow coordinate transformations that would require ugly code to fix - Bovy (UofT)

        """
        if radec or lb or uvw:
            raise NotImplementedError("Orbits initialization with radec=True, lb=True, and uvw=True is not implemented; please initialize using an array of astropy SkyCoords instead")
        if _APY_LOADED and isinstance(vxvv,SkyCoord):
            # Convert entire SkyCoord to Galactocentric coordinates, then
            # proceed in regular fashion; makes use of the fact that Orbit
            # setup does the right processing even for an array of 
            # SkyCoords
            tmpo= Orbit(vxvv=vxvv,ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)
            vxvv= numpy.array(tmpo._orb.vxvv).T
            # Grab coordinate-transform params, we know these must be 
            # consistent with any explicitly given ones at this point, because
            # Orbit setup would have thrown an error otherwise
            ro= tmpo._ro
            zo= tmpo._orb._zo
            solarmotion= tmpo._orb._solarmotion
        self._orbits = []
        for coord in vxvv:
            if isinstance(coord, Orbit):
                self._orbits.append(copy.deepcopy(coord))
            else:
                orbit = Orbit(vxvv=coord,radec=radec,uvw=uvw,lb=lb,
                              ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)
                self._orbits.append(orbit)
        self._setup_unit_coord_parameters(ro,vo,zo,solarmotion)
        # Cross-checks: phase-space dimension
        if not numpy.all([o.phasedim() == self._orbits[0].phasedim() 
                          for o in self._orbits]):
            raise RuntimeError("All individual orbits in an Orbits class must have the same phase-space dimensionality")
        # Store all vxvv and make individual ones views
        self.vxvv= numpy.array([o._orb.vxvv for o in self._orbits])
        for ii in range(len(self)):
            self._orbits[ii]._orb.vxvv= self.vxvv[ii]

    def _setup_unit_coord_parameters(self,ro,vo,zo,solarmotion):
        if _APY_LOADED and isinstance(ro,units.Quantity):
            ro= ro.to(units.kpc).value
        if _APY_LOADED and isinstance(zo,units.Quantity):
            zo= zo.to(units.kpc).value
        if _APY_LOADED and isinstance(vo,units.Quantity):
            vo= vo.to(units.km/units.s).value
        if _APY_LOADED and isinstance(solarmotion,units.Quantity):
            solarmotion= solarmotion.to(units.km/units.s).value
        # If parameters are not set, grab them from the first orbit (should
        # be consistent anyway, checked later)
        if vo is None:
            self._vo= self._orbits[0]._orb._vo
            self._voSet= False
        else:
            self._vo= vo
            self._voSet= True
        if ro is None:
            self._ro= self._orbits[0]._orb._ro
            self._roSet= False
        else:
            self._ro= ro
            self._roSet= True
        if zo is None:
            self._zo= self._orbits[0]._orb._zo
        else:
            self._zo= zo
        if solarmotion is None:
            self._solarmotion= self._orbits[0]._orb._solarmotion
        elif isinstance(solarmotion,str):
            # Use a dummy orbit to parse this consistently with how it's done
            # in Orbit
            dummy_orbit= Orbit([1.,0.1,1.1,0.1,0.2,0.],solarmotion=solarmotion)
            self._solarmotion= dummy_orbit._orb._solarmotion
        else:
            self._solarmotion= numpy.array(solarmotion)
        # Cross-checks: unit- and coordinate-conversion parameters
        if not numpy.all([numpy.fabs(o._ro-self._orbits[0]._ro) < 1e-10
                          for o in self._orbits]):
            raise RuntimeError("All individual orbits in an Orbits class must have the same ro unit-conversion parameter")
        if not numpy.fabs(self._ro-self._orbits[0]._ro) < 1e-10:
            raise RuntimeError("All individual orbits in an Orbits class must have the same ro unit-conversion parameter as the main Orbits object")
        if not numpy.all([numpy.fabs(o._vo-self._orbits[0]._vo) < 1e-10
                         for o in self._orbits]):
            raise RuntimeError("All individual orbits in an Orbits class must have the same vo unit-conversion parameter")
        if not numpy.fabs(self._vo-self._orbits[0]._vo) < 1e-10:
            raise RuntimeError("All individual orbits in an Orbits class must have the same vo unit-conversion parameter as the main Orbits object")
        if self.dim() > 1:
            if not self._orbits[0]._orb._zo is None \
                    and not numpy.all([numpy.fabs(o._orb._zo
                                            -self._orbits[0]._orb._zo) < 1e-10
                                       for o in self._orbits]):
                raise RuntimeError("All individual orbits in an Orbits class must have the same zo solar offset")
            if not numpy.fabs(self._zo-self._orbits[0]._orb._zo) < 1e-10:
                raise RuntimeError("All individual orbits in an Orbits class must have the same zo solar offset as the main Orbits object")
        if self.dim() > 1:
            if not numpy.all([numpy.fabs(o._orb._solarmotion
                                    -self._orbits[0]._orb._solarmotion) < 1e-10
                              for o in self._orbits]):
                raise RuntimeError("All individual orbits in an Orbits class must have the same solar motion")
            if not numpy.all(numpy.fabs(self._solarmotion
                                  -self._orbits[0]._orb._solarmotion) < 1e-10):
                raise RuntimeError("All individual orbits in an Orbits class must have the same solar motion as the main Orbits object")
        # If a majority of input orbits have roSet or voSet, set ro/vo for all
        if numpy.sum([o._roSet for o in self._orbits]) >= len(self)/2. \
                or numpy.sum([o._voSet for o in self._orbits]) >= len(self)/2.:
            [o.turn_physical_on(vo=self._vo,ro=self._ro) for o in self._orbits]
            self._roSet= True
            self._voSet= True
        return None

    def __len__(self):
        return len(self._orbits)
    def dim(self):
        return self._orbits[0].dim()
    def phasedim(self):
        return self._orbits[0].phasedim()

    # (temporary) solution to the fact that some custom-implemented functions
    # are not yet completely implemented, so if they aren't complete, fall
    # back onto Orbit functions; need to make sure that NotImplementedError
    # is raised at the top-level of the incomplete function, before calling
    # any other incomplete function... (see R function for instance in 
    # 3cf76f180545acb0606b2556135e9390ce800377)
    def __getattribute__(self,attr):
        if callable(super(Orbits,self).__getattribute__(attr)) \
                and not attr == '_pot' and not attr == '_aA':
            def func(*args,**kwargs):
                try:
                    out= super(Orbits,self)\
                        .__getattribute__(attr)(*args,**kwargs)
                except NotImplementedError:
                    out= self.__getattr__(attr)(*args,**kwargs)
                return out
            return func
        else:
            return super(Orbits,self).__getattribute__(attr)

    def __getattr__(self, name):
        """
        NAME:

            __getattr__

        PURPOSE:

            get or evaluate an attribute for these Orbits

        INPUT:

            name - name of the attribute

        OUTPUT:

            if the attribute is callable, a function to evaluate the attribute for each Orbit; otherwise a list of attributes

        HISTORY:

            2018-10-13 - Written - Mathew Bub (UofT)

            2019-02-28 - Implement all plotting function - Bovy (UofT)

        """
        # Catch all plotting functions
        if 'plot' in name:
            def _plot(*args,**kwargs):
                for ii in range(len(self)):
                    line2d= \
                     self._orbits[ii].__getattribute__(name)(*args,**kwargs)[0]
                    kwargs['overplot']= True
                line2d.axes.autoscale(enable=True)
                _add_ticks()
                return None
            # Assign documentation
            _plot.__doc__= self._orbits[0].__getattribute__(name).__doc__
            return _plot
        attribute = getattr(Orbit(), name)
        if callable(attribute):
            return lambda *args, **kwargs: [
                getattr(orbit, name)(*args, **kwargs) for orbit in self._orbits
            ]
        else:
            return [getattr(orbit, name) for orbit in self.orbits]

    def __getitem__(self,key):
        """
        NAME:

            __getitem__

        PURPOSE:

            get a subset of this instance's orbits

        INPUT:

           key - slice

        OUTPUT:

           For single item: Orbit instance, for multiple items: another Orbits instance

        HISTORY:

            2018-12-31 - Written - Bovy (UofT)

        """
        if isinstance(key,int):
            if key < 0 : # negative indices
                key+= len(self)
            return copy.deepcopy(self._orbits[key])
        elif isinstance(key,slice):
            orbits_list= [copy.deepcopy(self._orbits[ii]) 
                          for ii in range(*key.indices(len(self)))]
            # Also transfer all attributes related to integration
            if hasattr(self,'orbit'):
                integrate_kwargs= {}
                integrate_kwargs['t']= self.t
                integrate_kwargs['_integrate_t_asQuantity']= \
                    self._integrate_t_asQuantity
                integrate_kwargs['orbit']= copy.deepcopy(self.orbit[key])
                integrate_kwargs['_pot']= self._pot
            else: integrate_kwargs= None
            return Orbits._from_slice(orbits_list,integrate_kwargs)

    @classmethod
    def _from_slice(cls,orbits_list,integrate_kwargs):
        out= cls(vxvv=orbits_list)
        # Also transfer all attributes related to integration
        if not integrate_kwargs is None:
            for kw in integrate_kwargs:
                out.__dict__[kw]= integrate_kwargs[kw]
        return out

############################ CUSTOM IMPLEMENTED ORBIT FUNCTIONS################
    def turn_physical_off(self):
        """
        NAME:

           turn_physical_off

        PURPOSE:

           turn off automatic returning of outputs in physical units

        INPUT:

           (none)

        OUTPUT:

           (none)

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        self._roSet= False
        self._voSet= False
        [o.turn_physical_off() for o in self._orbits]
        return None

    def turn_physical_on(self,ro=None,vo=None):
        """
        NAME:

           turn_physical_on

        PURPOSE:

           turn on automatic returning of outputs in physical units

        INPUT:

           ro= reference distance (kpc; can be Quantity)

           vo= reference velocity (km/s; can be Quantity)

        OUTPUT:

           (none)

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        self._roSet= True
        self._voSet= True
        if not ro is None:
            if _APY_LOADED and isinstance(ro,units.Quantity):
                ro= ro.to(units.kpc).value
            self._ro= ro
        if not vo is None:
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
            self._vo= vo
        [o.turn_physical_on(vo=self._vo,ro=self._ro) for o in self._orbits]
        return None

    def integrate(self,t,pot,method='symplec4_c',dt=None,numcores=_NUMCORES,
                  force_map=False):
        """
        NAME:

            integrate

        PURPOSE:

            integrate these Orbits with multiprocessing

        INPUT:

            t - list of times at which to output (0 has to be in this!) (can be Quantity)

            pot - potential instance or list of instances

            method = 'odeint' for scipy's odeint
                     'leapfrog' for a simple leapfrog implementation
                     'leapfrog_c' for a simple leapfrog implementation in C
                     'symplec4_c' for a 4th order symplectic integrator in C
                     'symplec6_c' for a 6th order symplectic integrator in C
                     'rk4_c' for a 4th-order Runge-Kutta integrator in C
                     'rk6_c' for a 6-th order Runge-Kutta integrator in C
                     'dopr54_c' for a 5-4 Dormand-Prince integrator in C
                     'dopr853_c' for a 8-5-3 Dormand-Prince integrator in C

            dt - if set, force the integrator to use this basic stepsize; must be an integer divisor of output stepsize (only works for the C integrators that use a fixed stepsize) (can be Quantity)

            numcores - number of cores to use for multiprocessing with force_map; default = 1

        OUTPUT:

            None (get the actual orbit using getOrbit())

        HISTORY:

            2018-10-13 - Written as parallel_map applied to regular Orbit integration - Mathew Bub (UofT)

            2018-12-26 - Written to use OpenMP C implementation - Bovy (UofT)

        """
        if method.lower() not in ['odeint', 'leapfrog', 'dop853', 'leapfrog_c',
                'symplec4_c', 'symplec6_c', 'rk4_c', 'rk6_c',
                'dopr54_c', 'dop853_c']:
            raise ValueError('{:s} is not a valid `method`'.format(method))
        pot= flatten_potential(pot)
        _check_potential_dim(self,pot)
        _check_consistent_units(self,pot)
        # Parse t
        if _APY_LOADED and isinstance(t,units.Quantity):
            self._integrate_t_asQuantity= True
            t= t.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
        else: self._integrate_t_asQuantity= False
        if _APY_LOADED and not dt is None and isinstance(dt,units.Quantity):
            dt= dt.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
        from galpy.potential import MWPotential
        if pot == MWPotential:
            warnings.warn("Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                          galpyWarning)
        if not _check_integrate_dt(t,dt):
            raise ValueError('dt input (integrator stepsize) for Orbits.integrate must be an integer divisor of the output stepsize')
        # Delete attributes for interpolation and rperi etc. determination
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        if self.dim() == 2:
            thispot= toPlanarPotential(pot)
        else:
            thispot= pot
        self.t= numpy.array(t)
        self._pot= thispot
        #First check that the potential has C
        if '_c' in method:
            if not ext_loaded or not _check_c(self._pot):
                if ('leapfrog' in method or 'symplec' in method):
                    method= 'leapfrog'
                else:
                    method= 'odeint'
                if not ext_loaded: # pragma: no cover
                    warnings.warn("Cannot use C integration because C extension not loaded (using %s instead)" % (method), galpyWarning)
                else:
                    warnings.warn("Cannot use C integration because some of the potentials are not implemented in C (using %s instead)" % (method), galpyWarning)
        # Now check that we aren't trying to integrate a dissipative force
        # with a symplectic integrator
        if _isDissipative(self._pot) and ('leapfrog' in method 
                                    or 'symplec' in method):
            if '_c' in method:
                method= 'dopr54_c'
            else:
                method= 'odeint'
            warnings.warn("Cannot use symplectic integration because some of the included forces are dissipative (using non-symplectic integrator %s instead)" % (method), galpyWarning)
        # Implementation with parallel_map in Python
        if not '_c' in method or not ext_loaded or force_map:
            # Must return each Orbit for its values to correctly update
            def integrate_for_map(orbit):
                orbit.integrate(t, self._pot, method=method, dt=dt)
                return orbit
            self._orbits = list(parallel_map(integrate_for_map, self._orbits,
                                             numcores=numcores))
            # Gather all into single self.orbit array
            self.orbit= numpy.array([self._orbits[ii]._orb.orbit
                                     for ii in range(len(self))])
        else:
            warnings.warn("Using C implementation to integrate orbits",
                          galpyWarningVerbose)
            if self.dim() == 1:
                out, msg= integrateLinearOrbit_c(self._pot,
                                                 numpy.copy(self.vxvv),
                                                 t,method,dt=dt)
            else:
                if self.phasedim() == 3 \
                   or self.phasedim() == 5:
                    #We hack this by putting in a dummy phi=0
                    vxvvs= numpy.pad(self.vxvv,((0,0),(0,1)),
                                     'constant',constant_values=0)
                else:
                    vxvvs= numpy.copy(self.vxvv)
                if self.dim() == 2:
                    out, msg= integratePlanarOrbit_c(self._pot,vxvvs,
                                                     t,method,dt=dt)
                else:
                    out, msg= integrateFullOrbit_c(self._pot,vxvvs,
                                                   t,method,dt=dt)

                if self.phasedim() == 3 \
                   or self.phasedim() == 5:
                    out= out[:,:,:-1]
            # Store orbit internally
            self.orbit= out
        # Also store per-orbit view of the orbit for __getattr__ funcs
        for ii in range(len(self)):
            self._orbits[ii]._orb.orbit= self.orbit[ii]
            self._orbits[ii]._orb.t= t
            self._orbits[ii]._orb._pot= pot
        return None

    @physical_conversion('energy')
    def E(self,*args,**kwargs):
        """
        NAME:

           E

        PURPOSE:

           calculate the energy

        INPUT:

           t - (optional) time at which to get the energy (can be Quantity)

           pot= Potential instance or list of such instances

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           energy [norb,nt]

        HISTORY:

           2019-03-01 - Written - Bovy (UofT)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        _check_consistent_units(self,kwargs.get('pot',None))
        if not 'pot' in kwargs or kwargs['pot'] is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbits or specify pot=")
            if 'pot' in kwargs and kwargs['pot'] is None:
                kwargs.pop('pot')          
        else:
            pot= kwargs.pop('pot')
        if self.dim() == 2:
            pot= toPlanarPotential(pot)
        if len(args) > 0:
            t= args[0]
        else:
            t= 0.
        #Get orbit
        thiso= self._call_internal(*args,**kwargs)
        onet= (len(thiso.shape) == 2)
        if onet:
            thiso= thiso[:,numpy.newaxis,:]
            t= numpy.atleast_1d(t)
        if self.phasedim() == 2:
            try:
                out= (evaluatelinearPotentials(pot,thiso[0],t=t,
                                               use_physical=False)\
                          +thiso[1]**2./2.).T
            except (ValueError,TypeError):
                out= (numpy.array([[evaluatelinearPotentials(\
                                    pot,thiso[0][ii][jj],t=t[ii],
                                    use_physical=False)
                                    for ii in range(len(thiso[0]))]
                                   for jj in range(len(self))])\
                          +(thiso[1]**2./2.).T)
        elif self.phasedim() == 3:
            try:
                out= (evaluateplanarPotentials(pot,thiso[0],t=t,
                                               use_physical=False)\
                          +thiso[1]**2./2.+thiso[2]**2./2.).T
            except (ValueError,TypeError):
                out= (numpy.array([[evaluateplanarPotentials(\
                                    pot,thiso[0][ii][jj],t=t[ii],
                                    use_physical=False)
                                    for ii in range(len(thiso[0]))]
                                   for jj in range(len(self))])
                      +(thiso[1]**2./2.+thiso[2]**2./2.).T)
        elif self.phasedim() == 4:
            try:
                out= (evaluateplanarPotentials(pot,thiso[0],t=t,
                                               phi=thiso[-1],
                                               use_physical=False)\
                          +thiso[1]**2./2.+thiso[2]**2./2.).T
            except (ValueError,TypeError):
                out= (numpy.array([[evaluateplanarPotentials(\
                                    pot,thiso[0][ii][jj],t=t[ii],
                                    phi=thiso[-1][ii][jj],
                                    use_physical=False)
                                    for ii in range(len(thiso[0]))]
                                   for jj in range(len(self))])
                                  +(thiso[1]**2./2.+thiso[2]**2./2.).T)
        elif self.phasedim() == 5:
            z= kwargs.get('_z',1.)*thiso[3] # For ER and Ez
            vz= kwargs.get('_vz',1.)*thiso[4] # For ER and Ez
            try:
                out= (evaluatePotentials(pot,thiso[0],z,t=t,
                                         use_physical=False)\
                          +thiso[1]**2./2.+thiso[2]**2./2.+vz**2./2.).T
            except (ValueError,TypeError):
                out= (numpy.array([[evaluatePotentials(\
                                    pot,thiso[0][ii][jj],
                                    z[ii][jj],
                                    t=t[ii],
                                    use_physical=False)
                                    for ii in range(len(thiso[0]))]
                                   for jj in range(len(self))])
                      +(thiso[1]**2./2.+thiso[2]**2./2.+vz**2./2.).T)
        elif self.phasedim() == 6:
            z= kwargs.get('_z',1.)*thiso[3] # For ER and Ez
            vz= kwargs.get('_vz',1.)*thiso[4] # For ER and Ez
            try:
                out= (evaluatePotentials(pot,thiso[0],z,t=t,
                                         phi=thiso[-1],
                                         use_physical=False)\
                          +thiso[1]**2./2.+thiso[2]**2./2.+vz**2./2.).T
            except (ValueError,TypeError):
                out= (numpy.array([[evaluatePotentials(\
                                    pot,thiso[0][ii][jj],
                                    z[ii][jj],
                                    t=t[ii],
                                    phi=thiso[-1][ii][jj],
                                    use_physical=False)
                                    for ii in range(len(thiso[0]))]
                                   for jj in range(len(self))])
                      +(thiso[1]**2./2.+thiso[2]**2./2.+vz**2./2.).T)
        if onet:
            return out[:,0]
        else:
            return out

    @physical_conversion('action')
    def L(self,*args,**kwargs):
        """
        NAME:

           L

        PURPOSE:

           calculate the angular momentum at time t

        INPUT:

           t - (optional) time at which to get the angular momentum (can be Quantity)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           angular momentum [norb,nt,3]

        HISTORY:

           2019-03-01 - Written - Bovy (UofT)

        """
        if self.dim() == 1:
            raise AttributeError("'linear Orbits have no angular momentum")
        #Get orbit
        thiso= self._call_internal(*args,**kwargs)
        if self.dim() == 2:
            return (thiso[0]*thiso[2]).T
        elif self.phasedim() == 5:
            raise AttributeError("You must track the azimuth to get the angular momentum of a 3D Orbit")
        else: # phasedim == 6
            vx= self.vx(*args,**kwargs)
            vy= self.vy(*args,**kwargs)
            vz= self.vz(*args,**kwargs)
            x= self.x(*args,**kwargs)
            y= self.y(*args,**kwargs)
            z= self.z(*args,**kwargs)
            out= numpy.zeros(x.shape+(3,))
            out[...,0]= y*vz-z*vy
            out[...,1]= z*vx-x*vz
            out[...,2]= x*vy-y*vx
            return out
        
    @physical_conversion('action')
    def Lz(self,*args,**kwargs):
        """
        NAME:

           Lz

        PURPOSE:

           calculate the z-component of the angular momentum at time t

        INPUT:

           t - (optional) time at which to get the angular momentum (can be Quantity)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           z-component of the angular momentum [norb,nt]

        HISTORY:

           2019-03-01 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        return (thiso[0]*thiso[2]).T

    @physical_conversion('energy')
    def ER(self,*args,**kwargs):
        """
        NAME:

           ER

        PURPOSE:

           calculate the radial energy

        INPUT:

           t - (optional) time at which to get the radial energy (can be Quantity)

           pot= Potential instance or list of such instances

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output (can be Quantity)

        OUTPUT:

           radial energy [norb,nt]

        HISTORY:

           2019-03-01 - Written - Bovy (UofT)

        """
        old_physical= kwargs.get('use_physical',None)
        kwargs['use_physical']= False
        kwargs['_z']= 0.
        kwargs['_vz']= 0.
        out= self.E(*args,**kwargs)
        if not old_physical is None:
            kwargs['use_physical']= old_physical
        else:
            kwargs.pop('use_physical')
        return out

    @physical_conversion('energy')
    def Ez(self,*args,**kwargs):
        """
        NAME:

           Ez

        PURPOSE:

           calculate the vertical energy

        INPUT:

           t - (optional) time at which to get the vertical energy (can be Quantity)

           pot= Potential instance or list of such instances

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output (can be Quantity)

        OUTPUT:

           vertical energy [norb,nt]

        HISTORY:

           2019-03-01 - Written - Bovy (UofT)

        """
        old_physical= kwargs.get('use_physical',None)
        kwargs['use_physical']= False
        tE= self.E(*args,**kwargs)
        kwargs['_z']= 0.
        kwargs['_vz']= 0.
        out= tE-self.E(*args,**kwargs)
        if not old_physical is None:
            kwargs['use_physical']= old_physical
        else:
            kwargs.pop('use_physical')
        return out

    @physical_conversion('energy')
    def Jacobi(self,*args,**kwargs):
        """
        NAME:

           Jacobi

        PURPOSE:

           calculate the Jacobi integral E - Omega L

        INPUT:

           t - (optional) time at which to get the Jacobi integral (can be Quantity)

           OmegaP= pattern speed (can be Quantity)
           
           pot= potential instance or list of such instances

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           Jacobi integral [norb,nt]

        HISTORY:

           2019-03-01 - Written - Bovy (UofT)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        _check_consistent_units(self,kwargs.get('pot',None))
        if not 'OmegaP' in kwargs or kwargs['OmegaP'] is None:
            OmegaP= 1.
            if not 'pot' in kwargs or kwargs['pot'] is None:
                try:
                    pot= self._pot
                except AttributeError:
                    raise AttributeError("Integrate orbit or specify pot=")
            else:
                pot= kwargs['pot']
            if isinstance(pot,list):
                for p in pot:
                    if hasattr(p,'OmegaP'):
                        OmegaP= p.OmegaP()
                        break
            else:
                if hasattr(pot,'OmegaP'):
                    OmegaP= pot.OmegaP()
            kwargs.pop('OmegaP',None)
        else:
            OmegaP= kwargs.pop('OmegaP')
        if _APY_LOADED:
            if isinstance(OmegaP,units.Quantity):
                OmegaP = OmegaP.to(units.km/units.s/units.kpc).value \
                    /bovy_conversion.freq_in_kmskpc(self._vo,self._ro)
        #Make sure you are not using physical coordinates
        old_physical= kwargs.get('use_physical',None)
        kwargs['use_physical']= False
        if not isinstance(OmegaP,(int,float)) and len(OmegaP) == 3:
            if isinstance(OmegaP,list): thisOmegaP= numpy.array(OmegaP)
            else: thisOmegaP= OmegaP
            out= self.E(*args,**kwargs)-numpy.einsum('i,jki->jk',thisOmegaP,
                                                     self.L(*args,**kwargs))
        else:
            out= self.E(*args,**kwargs)-OmegaP*self.Lz(*args,**kwargs)
        if not old_physical is None:
            kwargs['use_physical']= old_physical
        else:
            kwargs.pop('use_physical')
        return out

    def _setupaA(self,pot=None,type='staeckel',**kwargs):
        """
        NAME:
           _setupaA
        PURPOSE:
           set up an actionAngle module for this Orbit
        INPUT:
           pot - potential
           type= ('staeckel') type of actionAngle module to use
              1) 'adiabatic'
              2) 'staeckel'
              3) 'isochroneApprox'
              4) 'spherical'
        OUTPUT:
        HISTORY:
           2019-02-25 - Written based on OrbitTop._setupaA - Bovy (UofT)
        """
        if not pot is None: pot= flatten_potential(pot)
        if self.dim() == 2:
            # No reason to do Staeckel or isochroneApprox or spherical...
            type= 'adiabatic'
        elif self.dim() == 1:
            raise RuntimeError("Orbits action-angle methods are not supported for 1D orbits")
        if hasattr(self,'_aA'):
            if (not pot is None and pot != self._aAPot) \
                    or (not type is None and type != self._aAType) \
                    or ('delta' in kwargs and hasattr(self._aA,'_delta') 
                        and numpy.any(kwargs['delta'] != self._aA._delta)) \
                    or (not 'delta' in kwargs 
                        and hasattr(self,'_aA_delta_automagic')
                        and not self._aA_delta_automagic):
                for attr in list(self.__dict__):
                    if '_aA' in attr: delattr(self,attr)
            else:
                return None
        _check_consistent_units(self,pot)
        if pot is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
        self._aAPot= pot
        self._aAType= type
        #Setup
        if self._aAType.lower() == 'adiabatic':
            self._aA= actionAngle.actionAngleAdiabatic(pot=self._aAPot,
                                                       **kwargs)
        elif self._aAType.lower() == 'staeckel':
            # try to make sure this is not 0
            tz= self.z(use_physical=False)\
                +(numpy.fabs(self.z(use_physical=False)) < 1e-8) \
                * (2.*(self.z(use_physical=False) >= 0)-1.)*1e-10
            delta= kwargs.pop('delta',None)
            self._aA_delta_automagic= False
            if delta is None:
                self._aA_delta_automagic= True
                try:
                    delta= actionAngle.estimateDeltaStaeckel(\
                        self._aAPot,self.R(use_physical=False),
                        tz,no_median=True)
                except PotentialError as e:
                    if 'deriv' in str(e):
                        raise PotentialError('Automagic calculation of delta parameter for Staeckel approximation failed because the necessary second derivatives of the given potential are not implemented; set delta= explicitly (to a single value or an array with the same shape as the orbits')
                    elif 'non-axi' in str(e):
                        raise PotentialError('Automagic calculation of delta parameter for Staeckel approximation failed because the given potential is not axisymmetric; pass an axisymmetric potential instead')
                    else: #pragma: no cover
                        raise
            if numpy.all(delta < 1e-6):
                self._setupaA(pot=pot,type='spherical')
            else:
                self._aA= actionAngle.actionAngleStaeckel(pot=self._aAPot,
                                                          delta=delta,
                                                          **kwargs)
        elif self._aAType.lower() == 'isochroneapprox':
            from galpy.actionAngle import actionAngleIsochroneApprox
            self._aA= actionAngleIsochroneApprox(pot=self._aAPot,
                                                 **kwargs)
        elif self._aAType.lower() == 'spherical':
            self._aA= actionAngle.actionAngleSpherical(pot=self._aAPot,
                                                       **kwargs)
        return None

    def _setup_EccZmaxRperiRap(self,pot=None,**kwargs):
        """Internal function to compute e,zmax,rperi,rap and cache it for re-use"""
        self._setupaA(pot=pot,**kwargs)
        if hasattr(self,'_aA_ecc'): return None
        if self.dim() == 3:
            # try to make sure this is not 0
            tz= self.z(use_physical=False)\
                +(numpy.fabs(self.z(use_physical=False)) < 1e-8) \
                * (2.*(self.z(use_physical=False) >= 0)-1.)*1e-10
            tvz= self.vz(use_physical=False)
        elif self.dim() == 2:
            tz= numpy.zeros(len(self))
            tvz= numpy.zeros(len(self))
        # self.dim() == 1 error caught by _setupaA
        self._aA_ecc, self._aA_zmax, self._aA_rperi, self._aA_rap=\
            self._aA.EccZmaxRperiRap(self.R(use_physical=False),
                                     self.vR(use_physical=False),
                                     self.vT(use_physical=False),
                                     tz,tvz,
                                     use_physical=False)
        return None        

    def _setup_actionsFreqsAngles(self,pot=None,**kwargs):
        """Internal function to compute the actions, frequencies, and angles and cache them for re-use"""
        self._setupaA(pot=pot,**kwargs)
        if hasattr(self,'_aA_jr'): return None
        if self.dim() == 3:
            # try to make sure this is not 0
            tz= self.z(use_physical=False)\
                +(numpy.fabs(self.z(use_physical=False)) < 1e-8) \
                * (2.*(self.z(use_physical=False) >= 0)-1.)*1e-10
            tvz= self.vz(use_physical=False)
        elif self.dim() == 2:
            tz= numpy.zeros(len(self))
            tvz= numpy.zeros(len(self))
        # self.dim() == 1 error caught by _setupaA
        self._aA_jr, self._aA_jp, self._aA_jz, \
            self._aA_Or, self._aA_Op, self._aA_Oz, \
            self._aA_wr, self._aA_wp, self._aA_wz= \
               self._aA.actionsFreqsAngles(self.R(use_physical=False),
                                           self.vR(use_physical=False),
                                           self.vT(use_physical=False),
                                           tz,tvz,
                                           self.phi(use_physical=False),
                                           use_physical=False)
        return None

    def e(self,analytic=False,pot=None,**kwargs):
        """
        NAME:

           e

        PURPOSE:

           calculate the eccentricity, either numerically from the numerical orbit integration or using analytical means

        INPUT:

           analytic(= False) compute this analytically

           pot - potential to use for analytical calculation

           For 3D orbits different approximations for analytic=True are available (see the EccZmaxRperiRap method of actionAngle modules):

              type= ('staeckel') type of actionAngle module to use
              
                 1) 'adiabatic': assuming motion splits into R and z

                 2) 'staeckel': assuming motion splits into u and v of prolate spheroidal coordinate system, exact for Staeckel potentials (incl. all spherical potentials)

                 3) 'spherical': for spherical potentials, exact
              
              +actionAngle module setup kwargs for the corresponding actionAngle modules (actionAngleAdiabatic, actionAngleStaeckel, and actionAngleSpherical)

        OUTPUT:

           eccentricity [norb]

        HISTORY:

           2019-02-25 - Written - Bovy (UofT)

        """
        if analytic:
            self._setup_EccZmaxRperiRap(pot=pot,**kwargs)
            return self._aA_ecc
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        rs= self.r(self.t,use_physical=False)
        return (numpy.amax(rs,axis=-1)-numpy.amin(rs,axis=-1))\
            /(numpy.amax(rs,axis=-1)+numpy.amin(rs,axis=-1))

    @physical_conversion('position')
    def rap(self,analytic=False,pot=None,**kwargs):
        """
        NAME:

           rap

        PURPOSE:

           calculate the apocenter radius, either numerically from the numerical orbit integration or using analytical means

        INPUT:

           analytic(= False) compute this analytically

           pot - potential to use for analytical calculation

           For 3D orbits different approximations for analytic=True are available (see the EccZmaxRperiRap method of actionAngle modules):

              type= ('staeckel') type of actionAngle module to use
              
                 1) 'adiabatic': assuming motion splits into R and z

                 2) 'staeckel': assuming motion splits into u and v of prolate spheroidal coordinate system, exact for Staeckel potentials (incl. all spherical potentials)

                 3) 'spherical': for spherical potentials, exact
              
              +actionAngle module setup kwargs for the corresponding actionAngle modules (actionAngleAdiabatic, actionAngleStaeckel, and actionAngleSpherical)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           R_ap [norb]

        HISTORY:

           2019-02-25 - Written - Bovy (UofT)

        """
        if analytic:
            self._setup_EccZmaxRperiRap(pot=pot,**kwargs)
            return self._aA_rap
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        rs= self.r(self.t,use_physical=False)
        return numpy.amax(rs,axis=-1)

    @physical_conversion('position')
    def rperi(self,analytic=False,pot=None,**kwargs):
        """
        NAME:

           rperi

        PURPOSE:

           calculate the pericenter radius, either numerically from the numerical orbit integration or using analytical means

        INPUT:

           analytic(= False) compute this analytically

           pot - potential to use for analytical calculation

           For 3D orbits different approximations for analytic=True are available (see the EccZmaxRperiRap method of actionAngle modules):

              type= ('staeckel') type of actionAngle module to use
              
                 1) 'adiabatic': assuming motion splits into R and z

                 2) 'staeckel': assuming motion splits into u and v of prolate spheroidal coordinate system, exact for Staeckel potentials (incl. all spherical potentials)

                 3) 'spherical': for spherical potentials, exact
              
              +actionAngle module setup kwargs for the corresponding actionAngle modules (actionAngleAdiabatic, actionAngleStaeckel, and actionAngleSpherical)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           R_peri [norb]

        HISTORY:

           2019-02-25 - Written - Bovy (UofT)

        """
        if analytic:
            self._setup_EccZmaxRperiRap(pot=pot,**kwargs)
            return self._aA_rperi
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        rs= self.r(self.t,use_physical=False)
        return numpy.amin(rs,axis=-1)

    @physical_conversion('position')
    def zmax(self,analytic=False,pot=None,**kwargs):
        """
        NAME:

           zmax

        PURPOSE:

           calculate the maximum vertical height, either numerically from the numerical orbit integration or using analytical means

        INPUT:

           analytic(= False) compute this analytically

           pot - potential to use for analytical calculation

           For 3D orbits different approximations for analytic=True are available (see the EccZmaxRperiRap method of actionAngle modules):

              type= ('staeckel') type of actionAngle module to use
              
                 1) 'adiabatic': assuming motion splits into R and z

                 2) 'staeckel': assuming motion splits into u and v of prolate spheroidal coordinate system, exact for Staeckel potentials (incl. all spherical potentials)

                 3) 'spherical': for spherical potentials, exact
              
              +actionAngle module setup kwargs for the corresponding actionAngle modules (actionAngleAdiabatic, actionAngleStaeckel, and actionAngleSpherical)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           Z_max [norb]

        HISTORY:

           2019-02-25 - Written - Bovy (UofT)

        """
        if analytic:
            self._setup_EccZmaxRperiRap(pot=pot,**kwargs)
            return self._aA_zmax
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        return numpy.amax(numpy.fabs(self.z(self.t,use_physical=False)),
                          axis=-1)

    @physical_conversion('action')
    def jr(self,pot=None,**kwargs):
        """
        NAME:

           jr

        PURPOSE:

           calculate the radial action

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           jr [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_jr

    @physical_conversion('action')
    def jp(self,pot=None,**kwargs):
        """
        NAME:

           jp

        PURPOSE:

           calculate the azimuthal action

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           jp [norb]

        HISTORY:

           2019-02-26 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_jp

    @physical_conversion('action')
    def jz(self,pot=None,**kwargs):
        """
        NAME:

           jz

        PURPOSE:

           calculate the vertical action

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           jz [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_jz

    @physical_conversion('angle')
    def wr(self,pot=None,**kwargs):
        """
        NAME:

           wr

        PURPOSE:

           calculate the radial angle

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           wr [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_wr

    @physical_conversion('angle')
    def wp(self,pot=None,**kwargs):
        """
        NAME:

           wp

        PURPOSE:

           calculate the azimuthal angle

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           wp [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_wp

    @physical_conversion('angle')
    def wz(self,pot=None,**kwargs):
        """
        NAME:

           wz

        PURPOSE:

           calculate the vertical angle

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           wz [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_wz

    @physical_conversion('time')
    def Tr(self,pot=None,**kwargs):
        """
        NAME:

           Tr

        PURPOSE:

           calculate the radial period

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           Tr [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return 2.*numpy.pi/self._aA_Or

    @physical_conversion('time')
    def Tp(self,pot=None,**kwargs):
        """
        NAME:

           Tp

        PURPOSE:

           calculate the azimuthal period

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           Tp [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return 2.*numpy.pi/self._aA_Op

    def TrTp(self,pot=None,**kwargs):
        """
        NAME:

           TrTp

        PURPOSE:

           the 'ratio' between the radial and azimuthal period Tr/Tphi*pi

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           Tr/Tp*pi [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_Op/self._aA_Or*numpy.pi
 
    @physical_conversion('time')
    def Tz(self,pot=None,**kwargs):
        """
        NAME:

           Tz

        PURPOSE:

           calculate the vertical period

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           Tz [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return 2.*numpy.pi/self._aA_Oz

    @physical_conversion('frequency')
    def Or(self,pot=None,**kwargs):
        """
        NAME:

           Or

        PURPOSE:

           calculate the radial frequency

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           Or [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_Or

    @physical_conversion('frequency')
    def Op(self,pot=None,**kwargs):
        """
        NAME:

           Op

        PURPOSE:

           calculate the azimuthal frequency

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           Op [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_Op

    @physical_conversion('frequency')
    def Oz(self,pot=None,**kwargs):
        """
        NAME:

           Oz

        PURPOSE:

           calculate the vertical frequency

        INPUT:

           pot - potential

           type= ('staeckel') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           Oz [norb]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_Oz

    @physical_conversion('time')
    def time(self,*args,**kwargs):
        """
        NAME:

           time

        PURPOSE:

           return the times at which the orbit is sampled

        INPUT:

           t - (default: integration times) time at which to get the time (for consistency reasons); default is to return the list of times at which the orbit is sampled

           ro= (Object-wide default) physical scale for distances to use to convert

           vo= (Object-wide default) physical scale for velocities to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           t(t)

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        if len(args) == 0:
            try:
                return self.t
            except AttributeError:
                return 0.
        else: return args[0]

    @physical_conversion('position')
    def R(self,*args,**kwargs):
        """
        NAME:

           R

        PURPOSE:

           return cylindrical radius at time t

        INPUT:

           t - (optional) time at which to get the radius (can be Quantity)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           R(t) [norb,nt]

        HISTORY:

           2019-02-01 - Written - Bovy (UofT)

        """
        return self._call_internal(*args,**kwargs)[0].T

    @physical_conversion('position')
    def r(self,*args,**kwargs):
        """
        NAME:

           r

        PURPOSE:

           return spherical radius at time t

        INPUT:

           t - (optional) time at which to get the radius

           ro= (Object-wide default) physical scale for distances to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           r(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        if self.dim() == 3:
            return numpy.sqrt(thiso[0]**2.+thiso[3]**2.).T
        else:
            return numpy.fabs(thiso[0]).T

    @physical_conversion('velocity')
    def vR(self,*args,**kwargs):
        """
        NAME:

           vR

        PURPOSE:

           return radial velocity at time t

        INPUT:

           t - (optional) time at which to get the radial velocity

           vo= (Object-wide default) physical scale for velocities to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vR(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        return self._call_internal(*args,**kwargs)[1].T

    @physical_conversion('velocity')
    def vT(self,*args,**kwargs):
        """
        NAME:

           vT

        PURPOSE:

           return tangential velocity at time t

        INPUT:

           t - (optional) time at which to get the tangential velocity

           vo= (Object-wide default) physical scale for velocities to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vT(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        return self._call_internal(*args,**kwargs)[2].T

    @physical_conversion('position')
    def z(self,*args,**kwargs):
        """
        NAME:

           z

        PURPOSE:

           return vertical height

        INPUT:

           t - (optional) time at which to get the vertical height

           ro= (Object-wide default) physical scale for distances to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           z(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        if self.dim() < 3:
            raise AttributeError("linear and planar orbits do not have z()")
        return self._call_internal(*args,**kwargs)[3].T

    @physical_conversion('velocity')
    def vz(self,*args,**kwargs):
        """
        NAME:

           vz

        PURPOSE:

           return vertical velocity

        INPUT:

           t - (optional) time at which to get the vertical velocity

           vo= (Object-wide default) physical scale for velocities to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vz(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        if self.dim() < 3:
            raise AttributeError("linear and planar orbits do not have vz()")
        return self._call_internal(*args,**kwargs)[4].T
        
    @physical_conversion('angle')
    def phi(self,*args,**kwargs):
        """
        NAME:

           phi

        PURPOSE:

           return azimuth

        INPUT:

           t - (optional) time at which to get the azimuth

        OUTPUT:

           phi(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        if self.phasedim() != 4 and self.phasedim() != 6:
            raise AttributeError("Orbits must track azimuth to use phi()")
        return self._call_internal(*args,**kwargs)[-1].T

    @physical_conversion('position')
    def x(self,*args,**kwargs):
        """
        NAME:

           x

        PURPOSE:

           return x

        INPUT:

           t - (optional) time at which to get x

           ro= (Object-wide default) physical scale for distances to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           x(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        if self.dim() == 1:
            return thiso[0].T
        elif self.phasedim()  != 4 and self.phasedim() != 6:
            raise AttributeError("Orbits must track azimuth to use x()")
        else:
            return (thiso[0]*numpy.cos(thiso[-1,:])).T

    @physical_conversion('position')
    def y(self,*args,**kwargs):
        """
        NAME:

           y

        PURPOSE:

           return y

        INPUT:

           t - (optional) time at which to get y

           ro= (Object-wide default) physical scale for distances to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           y(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        if self.phasedim()  != 4 and self.phasedim() != 6:
            raise AttributeError("Orbits must track azimuth to use y()")
        else:
            return (thiso[0]*numpy.sin(thiso[-1,:])).T

    @physical_conversion('velocity')
    def vx(self,*args,**kwargs):
        """
        NAME:

           vx

        PURPOSE:

           return x velocity at time t

        INPUT:

           t - (optional) time at which to get the velocity

           vo= (Object-wide default) physical scale for velocities to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vx(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        if self.dim() == 1:
            return thiso[1].T
        elif self.phasedim()  != 4 and self.phasedim() != 6:
            raise AttributeError("Orbits must track azimuth to use vx()")
        else:
            return (thiso[1]*numpy.cos(thiso[-1])
                    -thiso[2]*numpy.sin(thiso[-1])).T

    @physical_conversion('velocity')
    def vy(self,*args,**kwargs):
        """
        NAME:

           vy

        PURPOSE:

           return y velocity at time t

        INPUT:

           t - (optional) time at which to get the velocity

           vo= (Object-wide default) physical scale for velocities to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vy(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        if self.phasedim()  != 4 and self.phasedim() != 6:
            raise AttributeError("Orbits must track azimuth to use vy()")
        else:
            return (thiso[2]*numpy.cos(thiso[-1])
                    +thiso[1]*numpy.sin(thiso[-1])).T

    @physical_conversion('velocity')
    def vphi(self,*args,**kwargs):
        """
        NAME:

           vphi

        PURPOSE:

           return angular velocity

        INPUT:

           t - (optional) time at which to get the angular velocity

           vo= (Object-wide default) physical scale for velocities to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vphi(t) [norb,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        return (thiso[2]/thiso[0]).T

    @physical_conversion('angle_deg')
    def ra(self,*args,**kwargs):
        """
        NAME:

           ra

        PURPOSE:

           return the right ascension

        INPUT:

           t - (optional) time at which to get ra

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)

        OUTPUT:

           ra(t) [norb,nt] 

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'ra')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _radec(self,thiso,*args,**kwargs).T[0]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('angle_deg')
    def dec(self,*args,**kwargs):
        """
        NAME:

           dec

        PURPOSE:

           return the declination

        INPUT:

           t - (optional) time at which to get dec

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)

        OUTPUT:

           dec(t) [norb,nt] 

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'dec')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _radec(self,thiso,*args,**kwargs).T[1]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('angle_deg')
    def ll(self,*args,**kwargs):
        """
        NAME:

           ll

        PURPOSE:

           return Galactic longitude

        INPUT:

           t - (optional) time at which to get ll

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

        OUTPUT:

           l(t) [norb,nt] 

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'ll')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _lbd(self,thiso,*args,**kwargs).T[0].reshape(thiso_shape[1:]).T

    @physical_conversion('angle_deg')
    def bb(self,*args,**kwargs):
        """
        NAME:

           bb

        PURPOSE:

           return Galactic latitude

        INPUT:

           t - (optional) time at which to get bb

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

        OUTPUT:

           b(t) [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT

        """
        _check_roSet(self,kwargs,'bb')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _lbd(self,thiso,*args,**kwargs).T[1].reshape(thiso_shape[1:]).T

    @physical_conversion('position_kpc')
    def dist(self,*args,**kwargs):
        """
        NAME:

           dist

        PURPOSE:

           return distance from the observer in kpc

        INPUT:

           t - (optional) time at which to get dist

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

        OUTPUT:

           dist(t) in kpc [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'dist')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _lbd(self,thiso,*args,**kwargs).T[2].reshape(thiso_shape[1:]).T

    @physical_conversion('proper-motion_masyr')
    def pmra(self,*args,**kwargs):
        """
        NAME:

           pmra

        PURPOSE:

           return proper motion in right ascension (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmra

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)    

           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)

        OUTPUT:

           pm_ra(t) in mas / yr [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'pmra')
        _check_voSet(self,kwargs,'pmra')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _pmrapmdec(self,thiso,*args,**kwargs).T[0]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('proper-motion_masyr')
    def pmdec(self,*args,**kwargs):
        """
        NAME:

           pmdec

        PURPOSE:

           return proper motion in declination (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmdec

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)

        OUTPUT:

           pm_dec(t) in mas/yr [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'pmdec')
        _check_voSet(self,kwargs,'pmdec')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _pmrapmdec(self,thiso,*args,**kwargs).T[1]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('proper-motion_masyr')
    def pmll(self,*args,**kwargs):
        """
        NAME:

           pmll

        PURPOSE:

           return proper motion in Galactic longitude (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmll

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)

        OUTPUT:

           pm_l(t) in mas/yr [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'pmll')
        _check_voSet(self,kwargs,'pmll')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _lbdvrpmllpmbb(self,thiso,*args,**kwargs).T[4]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('proper-motion_masyr')
    def pmbb(self,*args,**kwargs):
        """
        NAME:

           pmbb

        PURPOSE:

           return proper motion in Galactic latitude (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmbb

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)

        OUTPUT:

           pm_b(t) in mas/yr [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'pmbb')
        _check_voSet(self,kwargs,'pmbb')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _lbdvrpmllpmbb(self,thiso,*args,**kwargs).T[5]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('velocity_kms')
    def vlos(self,*args,**kwargs):
        """
        NAME:

           vlos

        PURPOSE:

           return the line-of-sight velocity (in km/s)

        INPUT:

           t - (optional) time at which to get vlos

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)

        OUTPUT:

           vlos(t) in km/s [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'vlos')
        _check_voSet(self,kwargs,'vlos')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _lbdvrpmllpmbb(self,thiso,*args,**kwargs).T[3]\
            .reshape(thiso_shape[1:]).T

    def vra(self,*args,**kwargs):
        """
        NAME:

           vra

        PURPOSE:

           return velocity in right ascension (km/s)

        INPUT:

           t - (optional) time at which to get vra (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           v_ra(t) in km/s [norb]

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'vra')
        _check_voSet(self,kwargs,'vra')
        dist= self.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            return units.Quantity(dist.to(units.kpc).value*_K*
                                  self.pmra(*args,**kwargs)\
                                      .to(units.mas/units.yr).value,
                                  unit=units.km/units.s)
        else:
            return dist*_K*self.pmra(*args,**kwargs)

    def vdec(self,*args,**kwargs):
        """
        NAME:

           vdec

        PURPOSE:

           return velocity in declination (km/s)

        INPUT:

           t - (optional) time at which to get vdec (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           v_dec(t) in km/s [norb]

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'vdec')
        _check_voSet(self,kwargs,'vdec')
        dist= self.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            return units.Quantity(dist.to(units.kpc).value*_K*
                                  self.pmdec(*args,**kwargs)\
                                      .to(units.mas/units.yr).value,
                                  unit=units.km/units.s)
        else:
            return dist*_K*self.pmdec(*args,**kwargs)

    def vll(self,*args,**kwargs):
        """
        NAME:

           vll

        PURPOSE:

           return the velocity in Galactic longitude (km/s)

        INPUT:

           t - (optional) time at which to get vll (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           v_l(t) in km/s [norb]

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'vll')
        _check_voSet(self,kwargs,'vll')
        dist= self.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            return units.Quantity(dist.to(units.kpc).value*_K*
                                  self.pmll(*args,**kwargs)\
                                      .to(units.mas/units.yr).value,
                                  unit=units.km/units.s)
        else:
            return dist*_K*self.pmll(*args,**kwargs)
        
    def vbb(self,*args,**kwargs):
        """
        NAME:

           vbb

        PURPOSE:

            return velocity in Galactic latitude (km/s)

        INPUT:

           t - (optional) time at which to get vbb (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           v_b(t) in km/s [norb]

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'vbb')
        _check_voSet(self,kwargs,'vbb')
        dist= self.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            return units.Quantity(dist.to(units.kpc).value*_K*
                                  self.pmbb(*args,**kwargs)\
                                      .to(units.mas/units.yr).value,
                                  unit=units.km/units.s)
        else:
            return dist*_K*self.pmbb(*args,**kwargs)

    @physical_conversion('position_kpc')
    def helioX(self,*args,**kwargs):
        """
        NAME:

           helioX

        PURPOSE:

           return Heliocentric Galactic rectangular x-coordinate (aka "X")

        INPUT:

           t - (optional) time at which to get X

           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

        OUTPUT:

           helioX(t) in kpc [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'helioX')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _helioXYZ(self,thiso,*args,**kwargs)[0]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('position_kpc')
    def helioY(self,*args,**kwargs):
        """
        NAME:

           helioY

        PURPOSE:

           return Heliocentric Galactic rectangular y-coordinate (aka "Y")

        INPUT:

           t - (optional) time at which to get Y

           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

        OUTPUT:

           helioY(t) in kpc [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'helioY')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _helioXYZ(self,thiso,*args,**kwargs)[1]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('position_kpc')
    def helioZ(self,*args,**kwargs):
        """
        NAME:

           helioZ

        PURPOSE:

           return Heliocentric Galactic rectangular z-coordinate (aka "Z")

        INPUT:

           t - (optional) time at which to get Z

           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

        OUTPUT:

           helioZ(t) in kpc [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'helioZ')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _helioXYZ(self,thiso,*args,**kwargs)[2]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('velocity_kms')
    def U(self,*args,**kwargs):
        """
        NAME:

           U

        PURPOSE:

           return Heliocentric Galactic rectangular x-velocity (aka "U")

        INPUT:

           t - (optional) time at which to get U

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)

        OUTPUT:

           U(t) in km/s [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'U')
        _check_voSet(self,kwargs,'U')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _XYZvxvyvz(self,thiso,*args,**kwargs)[3]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('velocity_kms')
    def V(self,*args,**kwargs):
        """
        NAME:

           V

        PURPOSE:

           return Heliocentric Galactic rectangular y-velocity (aka "V")

        INPUT:

           t - (optional) time at which to get U

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)

        OUTPUT:

           V(t) in km/s [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'V')
        _check_voSet(self,kwargs,'V')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _XYZvxvyvz(self,thiso,*args,**kwargs)[4]\
            .reshape(thiso_shape[1:]).T

    @physical_conversion('velocity_kms')
    def W(self,*args,**kwargs):
        """
        NAME:

           W

        PURPOSE:

           return Heliocentric Galactic rectangular z-velocity (aka "W")

        INPUT:

           t - (optional) time at which to get W

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         

           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)

        OUTPUT:

           W(t) in km/s [norb,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'W')
        _check_voSet(self,kwargs,'W')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _XYZvxvyvz(self,thiso,*args,**kwargs)[5]\
            .reshape(thiso_shape[1:]).T

    def SkyCoord(self,*args,**kwargs):
        """
        NAME:

           SkyCoord

        PURPOSE:

           return the positions and velocities as an astropy SkyCoord

        INPUT:

           t - (optional) time at which to get the position

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= distance in kpc corresponding to R=1. (default=Object-wide default)

           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)

        OUTPUT:

           SkyCoord(t) [norb,nt] 

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        kwargs.pop('quantity',None) # rm useless keyword to no conflict later
        _check_roSet(self,kwargs,'SkyCoord')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        radec= _radec(self,thiso,*args,**kwargs).T\
            .reshape((2,)+thiso_shape[1:])
        tdist= self.dist(quantity=False,*args,**kwargs).T
        if not _APY3: # pragma: no cover
            return coordinates.SkyCoord(radec[0]*units.degree,
                                        radec[1]*units.degree,
                                        distance=tdist*units.kpc,
                                        frame='icrs').T
        _check_voSet(self,kwargs,'SkyCoord')
        pmrapmdec= _pmrapmdec(self,thiso,*args,**kwargs).T\
            .reshape((2,)+thiso_shape[1:])
        tvlos= self.vlos(quantity=False,*args,**kwargs).T
        # Also return the Galactocentric frame used
        v_sun= coordinates.CartesianDifferential(\
            numpy.array([-self._solarmotion[0],
                       self._solarmotion[1]+self._vo,
                       self._solarmotion[2]])*units.km/units.s)
        return coordinates.SkyCoord(radec[0]*units.degree,
                                    radec[1]*units.degree,
                                    distance=tdist*units.kpc,
                                    pm_ra_cosdec=pmrapmdec[0]\
                                        *units.mas/units.yr,
                                    pm_dec=pmrapmdec[1]*units.mas/units.yr,
                                    radial_velocity=tvlos*units.km/units.s,
                                    frame='icrs',
                                    galcen_distance=\
                                        numpy.sqrt(self._ro**2.+self._zo**2.)\
                                        *units.kpc,
                                    z_sun=self._zo*units.kpc,
                                    galcen_v_sun=v_sun).T

    def _call_internal(self,*args,**kwargs):
        """
        NAME:
           _call_internal
        PURPOSE:
           return the orbits vector at time t (like OrbitTop's __call__)
        INPUT:
           t - desired time
        OUTPUT:
           [R,vR,vT,z,vz(,phi)] or [R,vR,vT(,phi)] depending on the orbit; shape = [phasedim,nt,norb]
        HISTORY:
           2019-02-01 - Started - Bovy (UofT)
           2019-02-18 - Written interpolation part - Bovy (UofT)
        """
        if len(args) == 0:
            return numpy.array(self.vxvv).T
        elif not hasattr(self,'t'):
            raise ValueError("Integrate instance before evaluating it at a specific time")
        else:
            t= args[0]
        # Parse t
        if _APY_LOADED and isinstance(t,units.Quantity):
            t= t.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
        elif '_integrate_t_asQuantity' in self.__dict__ \
                and self._integrate_t_asQuantity \
                and not numpy.all(t == self.t):
            # Not doing hasattr in above elif, bc currently slow due to overwrite of __getattribute__
            warnings.warn("You specified integration times as a Quantity, but are evaluating at times not specified as a Quantity; assuming that time given is in natural (internal) units (multiply time by unit to get output at physical time)",galpyWarning)
        if numpy.all(t == self.t): # Common case where one wants all integrated times
            return self.orbit.T
        elif isinstance(t,(int,float)) and hasattr(self,'t') \
                and t in list(self.t):
            return numpy.array(self.orbit[:,list(self.t).index(t),:]).T
        else:
            if isinstance(t,(int,float)): 
                nt= 1
                t= numpy.atleast_1d(t)
            else: 
                nt= len(t)
            try:
                self._setupOrbitInterp()
            except:
                out= numpy.zeros((self.phasedim(),nt,len(self)))
                for jj in range(nt):
                    try:
                        indx= list(self.t).index(t[jj])
                    except ValueError:
                        raise LookupError("Orbit interpolaton failed; integrate on finer grid")
                    out[:,jj]= self.orbit[:,indx].T
                return out #should always have nt > 1, bc otherwise covered by above
            out= numpy.empty((self.phasedim(),nt,len(self)))
            # Evaluating RectBivariateSpline on grid requires sorted arrays
            sindx= numpy.argsort(t)
            t= t[sindx]
            usindx= numpy.argsort(sindx) # to later unsort
            if self.phasedim() == 4 or self.phasedim() == 6:
                #Unpack interpolated x and y to R and phi
                x= self._orbInterp[0](t,self._orb_indx_4orbInterp)
                y= self._orbInterp[-1](t,self._orb_indx_4orbInterp)
                out[0]= numpy.sqrt(x*x+y*y)
                out[-1]= numpy.arctan2(y,x) % (2.*numpy.pi)
                for ii in range(1,self.phasedim()-1):
                    out[ii]= self._orbInterp[ii](t,self._orb_indx_4orbInterp)
            else:
                for ii in range(self.phasedim()):
                    out[ii]= self._orbInterp[ii](t,self._orb_indx_4orbInterp)
            if nt == 1:
                return out[:,0]
            else:
                t= t[usindx]
                return out[:,usindx]

    def _setupOrbitInterp(self):
        if hasattr(self,"_orbInterp"): return None
        # Setup one interpolation / phasedim, for all orbits simultaneously
        # First check that times increase
        if hasattr(self,"t"): #Orbit has been integrated
            if self.t[1] < self.t[0]: #must be backward
                sindx= numpy.argsort(self.t)
                # sort
                self.t= self.t[sindx]
                self.orbit= self.orbit[:,sindx]
                usindx= numpy.argsort(sindx) # to later unsort
        orbInterp= []
        orb_indx= numpy.arange(len(self))
        for ii in range(self.phasedim()):
            if (self.phasedim() == 4 or self.phasedim() == 6) and ii == 0:
                #Interpolate x and y rather than R and phi to avoid issues w/ phase wrapping
                orbInterp.append(interpolate.RectBivariateSpline(\
                        self.t,orb_indx,
                        (self.orbit[:,:,0]*numpy.cos(self.orbit[:,:,-1])).T,
                        ky=1,s=0.))
            elif (self.phasedim() == 4 or self.phasedim() == 6) and \
                    ii == self.phasedim()-1:
                orbInterp.append(interpolate.RectBivariateSpline(\
                        self.t,orb_indx,
                        (self.orbit[:,:,0]*numpy.sin(self.orbit[:,:,-1])).T,
                        ky=1,s=0.))
            else:
                orbInterp.append(interpolate.RectBivariateSpline(\
                        self.t,orb_indx,self.orbit[:,:,ii].T,ky=1,s=0.))
        self._orbInterp= orbInterp
        self._orb_indx_4orbInterp= orb_indx
        try: #unsort
            self.t= self.t[usindx]
            self.orbit= self.orbit[:,usindx]
        except: pass
        return None
