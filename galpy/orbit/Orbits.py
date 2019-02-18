import os
import copy
import warnings
import numpy
from .Orbit import Orbit, _check_integrate_dt, _check_potential_dim, \
    _check_consistent_units
from ..util import galpyWarning, galpyWarningVerbose
from ..util.bovy_conversion import physical_conversion
from ..util.multi import parallel_map
from ..util.bovy_plot import _add_ticks
from ..util import bovy_conversion
from ..potential import toPlanarPotential
from ..potential import flatten as flatten_potential
from ..potential.Potential import _check_c
from ..potential.DissipativeForce import _isDissipative
from .integrateLinearOrbit import integrateLinearOrbit_c, _ext_loaded
from .integratePlanarOrbit import integratePlanarOrbit_c
from .integrateFullOrbit import integrateFullOrbit_c
ext_loaded= _ext_loaded
try:
    from astropy.coordinates import SkyCoord
    _APY_LOADED = True
except ImportError:
    SkyCoord = None
    _APY_LOADED = False
if _APY_LOADED:
    from astropy import units
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
    # any other incomplete function... (see R function for instance)
    def __getattribute__(self,attr):
        if callable(super(Orbits,self).__getattribute__(attr)) \
                and not attr == '_pot':
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

        """
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
            if hasattr(self,'orbit'):
                integrated_orbits= copy.deepcopy(self.orbit[key])
            else: integrated_orbits= None
            return Orbits._from_slice(orbits_list,integrated_orbits)

    @classmethod
    def _from_slice(cls,orbits_list,integrated_orbits=None):
        out= cls(vxvv=orbits_list)
        if not integrated_orbits is None:
            out.orbits= integrated_orbits
        return out

############################ CUSTOM IMPLEMENTED ORBIT FUNCTIONS################
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
            #self._orb._integrate_t_asQuantity= True
            t= t.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
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

           R(t)

        HISTORY:

           2019-02-01 - Written - Bovy (UofT)

        """
        if len(args) == 0:
            return self._call_internal(*args,**kwargs)[0]
        else:
            raise NotImplementedError("Function not yet fully custom-implemented for Orbits")

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
        """
        if len(args) == 0:
            return numpy.array(self.vxvv).T
        else:
            raise NotImplementedError("Function not yet fully custom-implemented for Orbits")

    def plot(self,*args,**kwargs):
        """
        Like Orbit.plot but for Orbits, same exact calling sequence
        Written - 2018-12-19 - Bovy (UofT)"""
        for ii in range(len(self)):
            line2d= self._orbits[ii].plot(*args,**kwargs)[0]
            kwargs['overplot']= True
        line2d.axes.autoscale(enable=True)
        _add_ticks()
        return None

