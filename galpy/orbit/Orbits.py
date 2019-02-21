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
    from astropy import units, coordinates
    import astropy
    _APY3= astropy.__version__ > '3'
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
        return numpy.sqrt(thiso[0]**2.+thiso[3]**2.).T

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

