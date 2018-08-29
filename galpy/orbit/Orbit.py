import warnings
import numpy as nu
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
if _APY_LOADED:
    import astropy
    _APY3= astropy.__version__ > '3'
    from astropy.coordinates import SkyCoord, Galactocentric, \
        CartesianDifferential
_ASTROQUERY_LOADED= True
try:
    from astroquery.simbad import Simbad
except ImportError:
    _ASTROQUERY_LOADED= False
import galpy.util.bovy_coords as coords
from galpy.util.bovy_conversion import physical_conversion
from galpy.util import galpyWarning
from galpy.util import bovy_conversion
from galpy.util import config
_APY_UNITS= config.__config__.getboolean('astropy','astropy-units')
from .FullOrbit import FullOrbit
from .RZOrbit import RZOrbit
from .planarOrbit import planarOrbit, planarROrbit, \
    planarOrbitTop
from .linearOrbit import linearOrbit
from galpy.potential import flatten as flatten_potential
from galpy.potential import rl, _isNonAxi
_K=4.74047
if _APY_LOADED:
    vxvv_units= [units.kpc,units.km/units.s,units.km/units.s,
                 units.kpc,units.km/units.s,units.rad]
else:
    _APY_UNITS= False
class Orbit(object):
    """General orbit class representing an orbit"""
    def __init__(self,vxvv=None,uvw=False,lb=False,
                 radec=False,vo=None,ro=None,zo=None,
                 solarmotion=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize an Orbit instance

        INPUT:

           vxvv - initial conditions 
                  3D can be either

              1) in Galactocentric cylindrical coordinates [R,vR,vT(,z,vz,phi)]; can be Quantities

              2) astropy (>v3.0) SkyCoord that includes velocities (Note that this turns *on* physical output even if ro and vo are not given)

              3) [ra,dec,d,mu_ra, mu_dec,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (all J2000.0; mu_ra = mu_ra * cos dec); can be Quantities; ICRS frame

              4) [ra,dec,d,U,V,W] in [deg,deg,kpc,km/s,km/s,kms]; can be Quantities; ICRS frame

              5) (l,b,d,mu_l, mu_b, vlos) in [deg,deg,kpc,mas/yr,mas/yr,km/s) (all J2000.0; mu_l = mu_l * cos b); can be Quantities

              6) [l,b,d,U,V,W] in [deg,deg,kpc,km/s,km/s,kms]; can be Quantities

              7) Unspecified: assumed to be the Sun (equivalent to ``vxvv= [0,0,0,0,0,0]`` and ``radec=True``)

           5) and 6) also work when leaving out b and mu_b/W

        OPTIONAL INPUTS:

           radec= if True, input is 2) (or 3) above (Note that this turns *on* physical output even if ro and vo are not given)

           uvw= if True, velocities are UVW

           lb= if True, input is 4) or 5) above (Note that this turns *on* physical output even if ro and vo are not given)

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

           zo= offset toward the NGP of the Sun wrt the plane (kpc; can be Quantity; default = 25 pc)

           solarmotion= 'hogg' or 'dehnen', or 'schoenrich', or value in 
           [-U,V,W]; can be Quantity

        If ro and/or vo are specified, outputs involving distances or velocities (whether as instance methods or in plots) will by default be displayed in the physical coordinates implied by these scales. This can be overwritten for each individual method by using use_physical=False as a keyword for the method.

        OUTPUT:

           instance

        HISTORY:

           2010-07-20 - Written - Bovy (NYU)

        """
        # If you change the way an Orbit object is setup, also change each of
        # the methods that return Orbits
        if vxvv is None: # Assume one wants the Sun
            vxvv= [0.,0.,0.,0.,0.,0.]
            radec= True
        if _APY_LOADED and isinstance(ro,units.Quantity):
            ro= ro.to(units.kpc).value
        if _APY_LOADED and isinstance(zo,units.Quantity):
            zo= zo.to(units.kpc).value
        if _APY_LOADED and isinstance(vo,units.Quantity):
            vo= vo.to(units.km/units.s).value
        # if vxvv is SkyCoord, preferentially use its ro and zo
        if _APY_LOADED and isinstance(vxvv,SkyCoord):
            if not _APY3: # pragma: no cover
                raise ImportError('Orbit initialization using an astropy SkyCoord requires astropy >3.0')
            if zo is None and not vxvv.z_sun is None:
                zo= vxvv.z_sun.to(units.kpc).value
            elif not vxvv.z_sun is None:
                if nu.fabs(zo-vxvv.z_sun.to(units.kpc).value) > 1e-8:
                    raise ValueError("Orbit initialization's zo different from SkyCoord's z_sun; these should be the same for consistency")
            elif zo is None and not vxvv.galcen_distance is None:
                zo= 0.
            if ro is None and not vxvv.galcen_distance is None:
                ro= nu.sqrt(vxvv.galcen_distance.to(units.kpc).value**2.
                            -zo**2.)
            elif not vxvv.galcen_distance is None and \
                    nu.fabs(ro**2.-vxvv.galcen_distance.to(units.kpc).value**2.-zo**2.) > 1e-14:
                warnings.warn("Orbit's initialization normalization ro and zo are incompatible with SkyCoord's galcen_distance (should have galcen_distance^2 = ro^2 + zo^2)",galpyWarning)
        # If at this point ro/vo not set, use default from config
        if (_APY_LOADED and isinstance(vxvv,SkyCoord)) or radec or lb:
            if ro is None:
                ro= config.__config__.getfloat('normalization','ro')
            if vo is None:
                vo= config.__config__.getfloat('normalization','vo')
        # If at this point zo not set, use default
        if zo is None: zo= 0.025
        # if vxvv is SkyCoord, preferentially use its solarmotion
        if _APY_LOADED and isinstance(vxvv,SkyCoord) \
                and not vxvv.galcen_v_sun is None:
            sc_solarmotion= vxvv.galcen_v_sun.d_xyz.to(units.km/units.s).value
            sc_solarmotion[0]= -sc_solarmotion[0] # right->left
            sc_solarmotion[1]-= vo
            if solarmotion is None:
                solarmotion= sc_solarmotion
        # If at this point solarmotion not set, use default
        if solarmotion is None: solarmotion= 'schoenrich'
        if isinstance(solarmotion,str) and solarmotion.lower() == 'hogg':
            vsolar= nu.array([-10.1,4.0,6.7])
        elif isinstance(solarmotion,str) and solarmotion.lower() == 'dehnen':
            vsolar= nu.array([-10.,5.25,7.17])
        elif isinstance(solarmotion,str) \
                and solarmotion.lower() == 'schoenrich':
            vsolar= nu.array([-11.1,12.24,7.25])
        elif _APY_LOADED and isinstance(solarmotion,units.Quantity):
            vsolar= solarmotion.to(units.km/units.s).value
        else:
            vsolar= nu.array(solarmotion)
        # If both vxvv SkyCoord with vsun and solarmotion set, check the same
        if _APY_LOADED and isinstance(vxvv,SkyCoord) \
                and not vxvv.galcen_v_sun is None:
            if nu.any(nu.fabs(sc_solarmotion-vsolar) > 1e-8):
                raise ValueError("Orbit initialization's solarmotion parameter not compatible with SkyCoord's galcen_v_sun; these should be the same for consistency (this may be because you did not set vo; galcen_v_sun = solarmotion+vo for consistency)")
        # Now parse vxvv
        if _APY_LOADED and isinstance(vxvv,SkyCoord):
            galcen_v_sun= CartesianDifferential(\
                nu.array([-vsolar[0],vsolar[1]+vo,vsolar[2]])*units.km/units.s)
            gc_frame= Galactocentric(\
                galcen_distance=nu.sqrt(ro**2.+zo**2.)*units.kpc,
                z_sun=zo*units.kpc,galcen_v_sun=galcen_v_sun)
            vxvvg= vxvv.transform_to(gc_frame)
            vxvvg.representation= 'cylindrical'
            R= vxvvg.rho.to(units.kpc).value/ro
            phi= nu.pi-vxvvg.phi.to(units.rad).value
            z= vxvvg.z.to(units.kpc).value/ro
            try:
                vR= vxvvg.d_rho.to(units.km/units.s).value/vo
            except TypeError:
                raise TypeError("SkyCoord given to Orbit initialization does not have velocity data, which is required to setup an Orbit")
            vT= -(vxvvg.d_phi*vxvvg.rho)\
                .to(units.km/units.s,
                    equivalencies=units.dimensionless_angles()).value/vo
            vz= vxvvg.d_z.to(units.km/units.s).value/vo
            vxvv= [R,vR,vT,z,vz,phi]
        elif radec or lb:
            if radec:
                if _APY_LOADED and isinstance(vxvv[0],units.Quantity):
                    ra, dec= vxvv[0].to(units.deg).value, \
                        vxvv[1].to(units.deg).value
                else:
                    ra, dec= vxvv[0], vxvv[1]
                l,b= coords.radec_to_lb(ra,dec,degree=True,epoch=None)
                _extra_rot= True
            elif len(vxvv) == 4:
                l, b= vxvv[0], 0.
                _extra_rot= False
            else:
                l,b= vxvv[0],vxvv[1]
                _extra_rot= True
            if _APY_LOADED and isinstance(l,units.Quantity):
                l= l.to(units.deg).value
            if _APY_LOADED and isinstance(b,units.Quantity):
                b= b.to(units.deg).value
            if uvw:
                if _APY_LOADED and isinstance(vxvv[2],units.Quantity):
                    X,Y,Z= coords.lbd_to_XYZ(l,b,vxvv[2].to(units.kpc).value,
                                             degree=True)
                else:
                    X,Y,Z= coords.lbd_to_XYZ(l,b,vxvv[2],degree=True)
                vx= vxvv[3]
                vy= vxvv[4]
                vz= vxvv[5]
                if _APY_LOADED and isinstance(vx,units.Quantity):
                    vx= vx.to(units.km/units.s).value
                if _APY_LOADED and isinstance(vy,units.Quantity):
                    vy= vy.to(units.km/units.s).value
                if _APY_LOADED and isinstance(vz,units.Quantity):
                    vz= vz.to(units.km/units.s).value
            else:
                if radec:
                    if _APY_LOADED and isinstance(vxvv[3],units.Quantity):
                        pmra, pmdec= vxvv[3].to(units.mas/units.yr).value, \
                            vxvv[4].to(units.mas/units.yr).value
                    else:
                        pmra, pmdec= vxvv[3], vxvv[4]
                    pmll, pmbb= coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec,
                                                             degree=True,
                                                             epoch=None)
                    d, vlos= vxvv[2], vxvv[5]
                elif len(vxvv) == 4:
                    pmll, pmbb= vxvv[2], 0.
                    d, vlos= vxvv[1], vxvv[3]
                else:
                    pmll, pmbb= vxvv[3], vxvv[4]
                    d, vlos= vxvv[2], vxvv[5]
                if _APY_LOADED and isinstance(d,units.Quantity):
                    d= d.to(units.kpc).value
                if _APY_LOADED and isinstance(vlos,units.Quantity):
                    vlos= vlos.to(units.km/units.s).value
                if _APY_LOADED and isinstance(pmll,units.Quantity):
                    pmll= pmll.to(units.mas/units.yr).value
                if _APY_LOADED and isinstance(pmbb,units.Quantity):
                    pmbb= pmbb.to(units.mas/units.yr).value
                X,Y,Z,vx,vy,vz= coords.sphergal_to_rectgal(l,b,d,
                                                           vlos,pmll, pmbb,
                                                           degree=True)
            X/= ro
            Y/= ro
            Z/= ro
            vx/= vo
            vy/= vo
            vz/= vo
            vsun= nu.array([0.,1.,0.,])+vsolar/vo
            R, phi, z= coords.XYZ_to_galcencyl(X,Y,Z,Zsun=zo/ro,
                                               _extra_rot=_extra_rot)
            vR, vT,vz= coords.vxvyvz_to_galcencyl(vx,vy,vz,
                                                  R,phi,z,
                                                  vsun=vsun,
                                                  Xsun=1.,Zsun=zo/ro,
                                                  galcen=True,
                                                  _extra_rot=_extra_rot)
            if lb and len(vxvv) == 4: vxvv= [R,vR,vT,phi]
            else: vxvv= [R,vR,vT,z,vz,phi]
        # Parse vxvv if it consists of Quantities
        if _APY_LOADED and isinstance(vxvv[0],units.Quantity):
            # Need to set ro and vo, default if not specified
            if ro is None:
                ro= config.__config__.getfloat('normalization','ro')
            if vo is None:
                vo= config.__config__.getfloat('normalization','vo')
            new_vxvv= [vxvv[0].to(vxvv_units[0]).value/ro,
                       vxvv[1].to(vxvv_units[1]).value/vo]
            if len(vxvv) > 2:
                new_vxvv.append(vxvv[2].to(vxvv_units[2]).value/vo)
            if len(vxvv) == 4:
                new_vxvv.append(vxvv[3].to(vxvv_units[5]).value)
            elif len(vxvv) > 4:
                new_vxvv.append(vxvv[3].to(vxvv_units[3]).value/ro)
                new_vxvv.append(vxvv[4].to(vxvv_units[4]).value/vo)
                if len(vxvv) == 6:
                    new_vxvv.append(vxvv[5].to(vxvv_units[5]).value)
            vxvv= new_vxvv
        if len(vxvv) == 2:
            self._orb= linearOrbit(vxvv=vxvv,
                                   ro=ro,vo=vo)
        elif len(vxvv) == 3:
            self._orb= planarROrbit(vxvv=vxvv,
                                    ro=ro,vo=vo,zo=zo,solarmotion=vsolar)
        elif len(vxvv) == 4:
            self._orb= planarOrbit(vxvv=vxvv,
                                    ro=ro,vo=vo,zo=zo,solarmotion=vsolar)
        elif len(vxvv) == 5:
            self._orb= RZOrbit(vxvv=vxvv,
                                    ro=ro,vo=vo,zo=zo,solarmotion=vsolar)
        elif len(vxvv) == 6:
            self._orb= FullOrbit(vxvv=vxvv,
                                    ro=ro,vo=vo,zo=zo,solarmotion=vsolar)
        #Store for actionAngle conversions
        if vo is None:
            self._vo= config.__config__.getfloat('normalization','vo')
            self._voSet= False
        else:
            self._vo= vo
            self._voSet= True
        if ro is None:
            self._ro= config.__config__.getfloat('normalization','ro')
            self._roSet= False
        else:
            self._ro= ro
            self._roSet= True
        return None

    def setphi(self,phi):
        """

        NAME:

           setphi

        PURPOSE:

           set initial azimuth

        INPUT:

           phi - desired azimuth

        OUTPUT:

           (none)

        HISTORY:

           2010-08-01 - Written - Bovy (NYU)

        BUGS:

           Should perform check that this orbit has phi

        """
        if len(self._orb.vxvv) == 2:
            raise AttributeError("One-dimensional orbit has no azimuth")
        elif len(self._orb.vxvv) == 3:
            #Upgrade
            vxvv= [self._orb.vxvv[0],self._orb.vxvv[1],self._orb.vxvv[2],phi]
            self._orb= planarOrbit(vxvv=vxvv)
        elif len(self._orb.vxvv) == 4:
            self._orb.vxvv[-1]= phi
        elif len(self._orb.vxvv) == 5:
            #Upgrade
            vxvv= [self._orb.vxvv[0],self._orb.vxvv[1],self._orb.vxvv[2],
                   self._orb.vxvv[3],self._orb.vxvv[4],phi]
            self._orb= FullOrbit(vxvv=vxvv)
        elif len(self._orb.vxvv) == 6:
            self._orb.vxvv[-1]= phi

    def dim(self):
        """
        NAME:

           dim

        PURPOSE:

           return the dimension of the problem

        INPUT:

           (none)

        OUTPUT:

           dimension

        HISTORY:

           2011-02-03 - Written - Bovy (NYU)

        """
        if len(self._orb.vxvv) == 2:
            return 1
        elif len(self._orb.vxvv) == 3 or len(self._orb.vxvv) == 4:
            return 2
        elif len(self._orb.vxvv) == 5 or len(self._orb.vxvv) == 6:
            return 3

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

           2014-06-17 - Written - Bovy (IAS)

        """
        self._roSet= False
        self._voSet= False
        self._orb.turn_physical_off()

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

           2016-01-19 - Written - Bovy (UofT)

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
        self._orb.turn_physical_on(ro=ro,vo=vo)

    def integrate(self,t,pot,method='symplec4_c',dt=None):
        """
        NAME:

           integrate

        PURPOSE:

           integrate the orbit

        INPUT:

           t - list of times at which to output (0 has to be in this!) (can be Quantity)

           pot - potential instance or list of instances

           method= 'odeint' for scipy's odeint
                   'leapfrog' for a simple leapfrog implementation
                   'leapfrog_c' for a simple leapfrog implementation in C
                   'symplec4_c' for a 4th order symplectic integrator in C
                   'symplec6_c' for a 6th order symplectic integrator in C
                   'rk4_c' for a 4th-order Runge-Kutta integrator in C
                   'rk6_c' for a 6-th order Runge-Kutta integrator in C
                   'dopr54_c' for a Dormand-Prince integrator in C (generally the fastest)

           dt= (None) if set, force the integrator to use this basic stepsize; must be an integer divisor of output stepsize (only works for the C integrators that use a fixed stepsize) (can be Quantity)

        OUTPUT:

           (none) (get the actual orbit using getOrbit()

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

           2015-06-28 - Added dt keyword - Bovy (IAS)

        """
        pot= flatten_potential(pot)
        _check_potential_dim(self,pot)
        _check_consistent_units(self,pot)
        # Parse t
        if _APY_LOADED and isinstance(t,units.Quantity):
            self._orb._integrate_t_asQuantity= True
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
            raise ValueError('dt input (integrator stepsize) for Orbit.integrate must be an integer divisor of the output stepsize')
        self._orb.integrate(t,pot,method=method,dt=dt)

    def integrate_dxdv(self,dxdv,t,pot,method='dopr54_c',
                       rectIn=False,rectOut=False):
        """
        NAME:

           integrate_dxdv

        PURPOSE:

           integrate the orbit and a small area of phase space

        INPUT:

           dxdv - [dR,dvR,dvT,dphi]

           t - list of times at which to output (0 has to be in this!) (can be Quantity)

           pot - potential instance or list of instances

           method= 'odeint' for scipy's odeint

                   'rk4_c' for a 4th-order Runge-Kutta integrator in C

                   'rk6_c' for a 6-th order Runge-Kutta integrator in C

                   'dopr54_c' for a Dormand-Prince integrator in C (generally the fastest)

                   'dopr54_c' is recommended, odeint is *not* recommended


           rectIn= (False) if True, input dxdv is in rectangular coordinates

           rectOut= (False) if True, output dxdv (that in orbit_dxdv) is in rectangular coordinates

        OUTPUT:

           (none) (get the actual orbit using getOrbit_dxdv(), the orbit that is integrated alongside with dxdv is stored as usual, any previous regular orbit integration will be erased!)

        HISTORY:

           2010-10-17 - Written - Bovy (IAS)

           2014-06-29 - Added rectIn and rectOut - Bovy (IAS)

        """
        pot= flatten_potential(pot)
        _check_potential_dim(self,pot)
        _check_consistent_units(self,pot)
        # Parse t
        if _APY_LOADED and isinstance(t,units.Quantity):
            self._orb._integrate_t_asQuantity= True
            t= t.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
        self._orb.integrate_dxdv(dxdv,t,pot,method=method,
                                 rectIn=rectIn,rectOut=rectOut)

    def reverse(self):
        """
        NAME:

           reverse

        PURPOSE:

           reverse an already integrated orbit (that is, make it go from end to beginning in t=0 to tend)

        INPUT:

           (none)

        OUTPUT:

           (none)

        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        sortindx = list(range(len(self._orb.t)))
        sortindx.sort(key=lambda x: self._orb.t[x],reverse=True)
        for ii in range(self._orb.orbit.shape[1]):
            self._orb.orbit[:,ii]= self._orb.orbit[sortindx,ii]
        return None

    def flip(self,inplace=False):
        """
        NAME:

           flip

        PURPOSE:

           'flip' an orbit's initial conditions such that the velocities are minus the original velocities; useful for quick backward integration; returns a new Orbit instance

        INPUT:

           inplace= (False) if True, flip the orbit in-place, that is, without returning a new instance and also flip the velocities of the integrated orbit (if it exists)

        OUTPUT:

           Orbit instance that has the velocities of the current orbit flipped (inplace=False) or just flips all velocities of current instance (inplace=True)

        HISTORY:

           2014-06-17 - Written - Bovy (IAS)

           2016-07-21 - Added inplace keyword - Bovy (UofT)

        """
        if inplace:
            self._orb.vxvv[1]= -self._orb.vxvv[1]
            if len(self._orb.vxvv) > 2:
                self._orb.vxvv[2]= -self._orb.vxvv[2]
            if len(self._orb.vxvv) > 4:
                self._orb.vxvv[4]= -self._orb.vxvv[4]
            if hasattr(self._orb,'orbit'):
                self._orb.orbit[:,1]= -self._orb.orbit[:,1]
                if len(self._orb.vxvv) > 2:
                    self._orb.orbit[:,2]= -self._orb.orbit[:,2]
                if len(self._orb.vxvv) > 4:
                    self._orb.orbit[:,4]= -self._orb.orbit[:,4]
                if hasattr(self._orb,"_orbInterp"):
                    delattr(self._orb,"_orbInterp")
            return None
        orbSetupKwargs= {'ro':None,
                         'vo':None,
                         'zo':self._orb._zo,
                         'solarmotion':self._orb._solarmotion}
        if self._orb._roSet:
            orbSetupKwargs['ro']= self._orb._ro
        if self._orb._voSet:
            orbSetupKwargs['vo']= self._orb._vo
        if len(self._orb.vxvv) == 2:
            return Orbit(vxvv= [self._orb.vxvv[0],-self._orb.vxvv[1]],
                         **orbSetupKwargs)
        elif len(self._orb.vxvv) == 3:
            return Orbit(vxvv=[self._orb.vxvv[0],-self._orb.vxvv[1],
                               -self._orb.vxvv[2]],**orbSetupKwargs)
        elif len(self._orb.vxvv) == 4:
            return Orbit(vxvv=[self._orb.vxvv[0],-self._orb.vxvv[1],
                               -self._orb.vxvv[2],self._orb.vxvv[3]],
                         **orbSetupKwargs)
        elif len(self._orb.vxvv) == 5:
            return Orbit(vxvv=[self._orb.vxvv[0],-self._orb.vxvv[1],
                               -self._orb.vxvv[2],self._orb.vxvv[3],
                               -self._orb.vxvv[4]],**orbSetupKwargs)
        elif len(self._orb.vxvv) == 6:
            return Orbit(vxvv= [self._orb.vxvv[0],-self._orb.vxvv[1],
                                -self._orb.vxvv[2],self._orb.vxvv[3],
                                -self._orb.vxvv[4],self._orb.vxvv[5]],
                         **orbSetupKwargs)
        
    def getOrbit(self):
        """

        NAME:

           getOrbit

        PURPOSE:

           return a previously calculated orbit

        INPUT:

           (none)

        OUTPUT:

           array orbit[nt,nd]

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._orb.getOrbit()

    def getOrbit_dxdv(self):
        """

        NAME:

           getOrbit_dxdv

        PURPOSE:

           return a previously calculated integration of a small phase-space volume (with integrate_dxdv)

        INPUT:

           (none)

        OUTPUT:

           array orbit[nt,nd*2] with for each t the dxdv vector

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._orb.getOrbit_dxdv()

    def fit(self,vxvv,vxvv_err=None,pot=None,radec=False,lb=False,
            customsky=False,lb_to_customsky=None,pmllpmbb_to_customsky=None,
            tintJ=10,ntintJ=1000,integrate_method='dopr54_c',
            **kwargs):
        """
        NAME:

           fit

        PURPOSE:

           fit an Orbit to data using the current orbit as the initial condition

        INPUT:

           vxvv - [:,6] array of positions and velocities along the orbit (if not lb=True or radec=True, these need to be in natural units [/ro,/vo], cannot be Quantities)

           vxvv_err= [:,6] array of errors on positions and velocities along the orbit (if None, these are set to 0.01) (if not lb=True or radec=True, these need to be in natural units [/ro,/vo], cannot be Quantities)

           pot= Potential to fit the orbit in

           Keywords related to the input data:

               radec= if True, input vxvv and vxvv are [ra,dec,d,mu_ra, mu_dec,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (all J2000.0; mu_ra = mu_ra * cos dec); the attributes of the current Orbit are used to convert between these coordinates and Galactocentric coordinates

               lb= if True, input vxvv and vxvv are [long,lat,d,mu_ll, mu_bb,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (mu_ll = mu_ll * cos lat); the attributes of the current Orbit are used to convert between these coordinates and Galactocentric coordinates

               customsky= if True, input vxvv and vxvv_err are [custom long,custom lat,d,mu_customll, mu_custombb,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (mu_ll = mu_ll * cos lat) where custom longitude and custom latitude are a custom set of sky coordinates (e.g., ecliptic) and the proper motions are also expressed in these coordinats; you need to provide the functions lb_to_customsky and pmllpmbb_to_customsky to convert to the custom sky coordinates (these should have the same inputs and outputs as lb_to_radec and pmllpmbb_to_pmrapmdec); the attributes of the current Orbit are used to convert between these coordinates and Galactocentric coordinates

               obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                                      (in kpc and km/s; entries can be Quantity) (default=Object-wide default)
                                      Cannot be an Orbit instance with the orbit of the reference point, as w/ the ra etc. functions
                                      Y is ignored and always assumed to be zero

                ro= distance in kpc corresponding to R=1. (default: taken from object; can be Quantity)

                vo= velocity in km/s corresponding to v=1. (default: taken from object; can be Quantity)

                lb_to_customsky= function that converts l,b,degree=False to the custom sky coordinates (like lb_to_radec); needs to be given when customsky=True

                pmllpmbb_to_customsky= function that converts pmll,pmbb,l,b,degree=False to proper motions in the custom sky coordinates (like pmllpmbb_to_pmrapmdec); needs to be given when customsky=True

           Keywords related to the orbit integrations:

               tintJ= (default: 10) time to integrate orbits for fitting the orbit (can be Quantity)

               ntintJ= (default: 1000) number of time-integration points

               integrate_method= (default: 'dopr54_c') integration method to use

           disp= (False) display the optimizer's convergence message

        OUTPUT:

           max of log likelihood

        HISTORY:

           2014-06-17 - Written - Bovy (IAS)

        """
        pot= flatten_potential(pot)
        _check_potential_dim(self,pot)
        _check_consistent_units(self,pot)
        return self._orb.fit(vxvv,vxvv_err=vxvv_err,pot=pot,
                             radec=radec,lb=lb,
                             customsky=customsky,
                             lb_to_customsky=lb_to_customsky,
                             pmllpmbb_to_customsky=pmllpmbb_to_customsky,
                             tintJ=tintJ,ntintJ=ntintJ,
                             integrate_method=integrate_method,
                             **kwargs)

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

           energy

        HISTORY:

           2010-09-15 - Written - Bovy (NYU)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        _check_consistent_units(self,kwargs.get('pot',None))
        return self._orb.E(*args,**kwargs)

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

           angular momentum

        HISTORY:

           2010-09-15 - Written - Bovy (NYU)

        """
        return self._orb.L(*args,**kwargs)

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

           z-component of the angular momentum

        HISTORY:

           2018-08-29 - Written - Bovy (UofT)

        """
        return self._orb.Lz(*args,**kwargs)

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

           radial energy

        HISTORY:

           2013-11-30 - Written - Bovy (IAS)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        _check_consistent_units(self,kwargs.get('pot',None))
        return self._orb.ER(*args,**kwargs)

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

           vertical energy

        HISTORY:

           2013-11-30 - Written - Bovy (IAS)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        _check_consistent_units(self,kwargs.get('pot',None))
        return self._orb.Ez(*args,**kwargs)

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

           Jacobi integral

        HISTORY:

           2011-04-18 - Written - Bovy (NYU)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        _check_consistent_units(self,kwargs.get('pot',None))
        out= self._orb.Jacobi(*args,**kwargs)
        if not isinstance(out,float) and len(out) == 1: return out[0]
        else: return out

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

           eccentricity

        HISTORY:

           2010-09-15 - Written - Bovy (NYU)

           2017-12-25 - Added Staeckel approximation and made that the default - Bovy (UofT)

        """

        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        return self._orb.e(analytic=analytic,pot=pot,**kwargs)

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

           R_ap

        HISTORY:

           2010-09-20 - Written - Bovy (NYU)

           2017-12-25 - Added Staeckel approximation and made that the default - Bovy (UofT)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        return self._orb.rap(analytic=analytic,pot=pot,**kwargs)

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

           R_peri

        HISTORY:

           2010-09-20 - Written - Bovy (NYU)

           2017-12-25 - Added Staeckel approximation and made that the default - Bovy (UofT)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        return self._orb.rperi(analytic=analytic,pot=pot,**kwargs)

    @physical_conversion('position')
    def rguiding(self,*args,**kwargs):
        """
        NAME:

           rguiding

        PURPOSE:

           calculate the guiding-center radius (the radius of a circular orbit with the same angular momentum)

        INPUT:

           pot= potential instance or list of such instances

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           R_guiding

        HISTORY:

           2018-08-29 - Written as thin wrapper around Potential.rl - Bovy (UofT)

        """
        pot= kwargs.get('pot',None)
        if pot is None:
            raise RuntimeError("You need to specify the potential as pot= to compute the guiding-center radius")
        flatten_potential(pot)
        if _isNonAxi(pot):
            raise RuntimeError('Potential given to rguiding is non-axisymmetric, but rguiding requires an axisymmetric potential')
        _check_consistent_units(self,pot)
        Lz= self.Lz(*args,use_physical=False)
        return nu.array([rl(pot,lz,use_physical=False) for lz in Lz])

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

           Z_max

        HISTORY:

           2010-09-20 - Written - Bovy (NYU)

           2017-12-25 - Added Staeckel approximation and made that the default - Bovy (UofT)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        return self._orb.zmax(analytic=analytic,pot=pot,**kwargs)

    def resetaA(self,pot=None,type=None):
        """
        NAME:

           resetaA

        PURPOSE:

           re-set up an actionAngle module for this Orbit

        INPUT:

           (none)

        OUTPUT:

           True if reset happened, False otherwise

        HISTORY:

           2014-01-06 - Written - Bovy (IAS)

        """
        try:
            delattr(self._orb,'_aA')
        except AttributeError:
            return False
        else:
            return True

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

           jr

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(self._orb._aA(self(),use_physical=False)[0])
        else:
            return float(self._orb._aA(self,use_physical=False)[0])

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

           jp

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(self._orb._aA(self(),use_physical=False)[1])
        else:
            return float(self._orb._aA(self,use_physical=False)[1])

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

           jz

        HISTORY:

           2012-06-01 - Written - Bovy (IAS)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(self._orb._aA(self(),use_physical=False)[2])
        else:
            return float(self._orb._aA(self,use_physical=False)[2])

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

           wr

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(self._orb._aA.actionsFreqsAngles(self(),
                                                    use_physical=False)[6][0])
        else:
            return float(self._orb._aA.actionsFreqsAngles(self,
                                                    use_physical=False)[6][0])

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

           wp

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(self._orb._aA.actionsFreqsAngles(self(),
                                                    use_physical=False)[7][0])
        else:
            return float(self._orb._aA.actionsFreqsAngles(self,
                                                    use_physical=False)[7][0])

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

           wz

        HISTORY:

           2012-06-01 - Written - Bovy (IAS)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(self._orb._aA.actionsFreqsAngles(self(),
                                                    use_physical=False)[8][0])
        else:
            return float(self._orb._aA.actionsFreqsAngles(self,
                                                    use_physical=False)[8][0])

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

           Tr

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(2.*nu.pi/self._orb._aA.actionsFreqs(self(),
                                                       use_physical=False)[3][0])
        else:
            return float(2.*nu.pi/self._orb._aA.actionsFreqs(self,
                                                       use_physical=False)[3][0])

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

           Tp

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(2.*nu.pi/self._orb._aA.actionsFreqs(self(),
                                                       use_physical=False)[4][0])
        else:
            return float(2.*nu.pi/self._orb._aA.actionsFreqs(self,
                                                       use_physical=False)[4][0])

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

           Tr/Tp*pi

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(self._orb._aA.actionsFreqs(self())[4][0]/self._orb._aA.actionsFreqs(self())[3][0]*nu.pi)
        else:
            return float(self._orb._aA.actionsFreqs(self)[4][0]/self._orb._aA.actionsFreqs(self)[3][0]*nu.pi)
 
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

           Tz

        HISTORY:

           2012-06-01 - Written - Bovy (IAS)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(2.*nu.pi/self._orb._aA.actionsFreqs(self(),
                                                       use_physical=False)[5][0])
        else:
            return float(2.*nu.pi/self._orb._aA.actionsFreqs(self,
                                                       use_physical=False)[5][0])

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

           Or

        HISTORY:

           2013-11-27 - Written - Bovy (IAS)

        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(self._orb._aA.actionsFreqs(self(),use_physical=False)[3][0])
        else:
            return float(self._orb._aA.actionsFreqs(self,use_physical=False)[3][0])

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

           Op

        HISTORY:

           2013-11-27 - Written - Bovy (IAS)
        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(self._orb._aA.actionsFreqs(self(),use_physical=False)[4][0])
        else:
            return float(self._orb._aA.actionsFreqs(self,use_physical=False)[4][0])

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

           Oz

        HISTORY:

           2013-11-27 - Written - Bovy (IAS)
        """
        if not pot is None: pot= flatten_potential(pot)
        _check_consistent_units(self,pot)
        self._orb._setupaA(pot=pot,**kwargs)
        if self._orb._aAType.lower() == 'isochroneapprox':
            return float(self._orb._aA.actionsFreqs(self(),use_physical=False)[5][0])
        else:
            return float(self._orb._aA.actionsFreqs(self,use_physical=False)[5][0])

    def time(self,*args,**kwargs):
        """
        NAME:

           t

        PURPOSE:

           return the times at which the orbit is sampled

        INPUT:

           t - (default: integration times) time at which to get the time (for consistency reasons); default is to return the list of times at which the orbit is sampled (can be Quantity)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           t(t)

        HISTORY:

           2014-06-11 - Written - Bovy (IAS)

        """
        return self._orb.time(*args,**kwargs)

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

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.R(*args,**kwargs)

    def r(self,*args,**kwargs):
        """
        NAME:

           r

        PURPOSE:

           return spherical radius at time t

        INPUT:

           t - (optional) time at which to get the radius (can be Quantity)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           r(t)

        HISTORY:

           2016-04-19 - Written - Bovy (UofT)

        """
        return self._orb.r(*args,**kwargs)

    def vR(self,*args,**kwargs):
        """
        NAME:

           vR

        PURPOSE:

           return radial velocity at time t

        INPUT:

           t - (optional) time at which to get the radial velocity

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vR(t)

        HISTORY:


           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.vR(*args,**kwargs)

    def vT(self,*args,**kwargs):
        """
        NAME:

           vT

        PURPOSE:

           return tangential velocity at time t

        INPUT:

           t - (optional) time at which to get the tangential velocity (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vT(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.vT(*args,**kwargs)

    def z(self,*args,**kwargs):
        """
        NAME:

           z

        PURPOSE:

           return vertical height

        INPUT:

           t - (optional) time at which to get the vertical height (can be Quantity)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           z(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.z(*args,**kwargs)

    def vz(self,*args,**kwargs):
        """
        NAME:

           vz

        PURPOSE:

           return vertical velocity

        INPUT:

           t - (optional) time at which to get the vertical velocity (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vz(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.vz(*args,**kwargs)

    def phi(self,*args,**kwargs):
        """
        NAME:

           phi

        PURPOSE:

           return azimuth

        INPUT:

           t - (optional) time at which to get the azimuth (can be Quantity)

        OUTPUT:

           phi(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.phi(*args,**kwargs)

    def vphi(self,*args,**kwargs):
        """
        NAME:

           vphi

        PURPOSE:

           return angular velocity

        INPUT:

           t - (optional) time at which to get the angular velocity (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vphi(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.vphi(*args,**kwargs)

    def x(self,*args,**kwargs):
        """
        NAME:

           x

        PURPOSE:

           return x

        INPUT:

           t - (optional) time at which to get x (can be Quantity)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           x(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        out= self._orb.x(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def y(self,*args,**kwargs):
        """
        NAME:

           y

        PURPOSE:

           return y

        INPUT:

           t - (optional) time at which to get y (can be Quantity)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           y(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        out= self._orb.y(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def vx(self,*args,**kwargs):
        """
        NAME:

           vx

        PURPOSE:

           return x velocity at time t

        INPUT:

           t - (optional) time at which to get the velocity (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vx(t)

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

        """
        out= self._orb.vx(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def vy(self,*args,**kwargs):
        """

        NAME:

           vy

        PURPOSE:

           return y velocity at time t

        INPUT:

           t - (optional) time at which to get the velocity (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vy(t)

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

        """
        out= self._orb.vy(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def ra(self,*args,**kwargs):
        """
        NAME:

           ra

        PURPOSE:

           return the right ascension

        INPUT:

           t - (optional) time at which to get ra (can be Quantity)

           obs=[X,Y,Z] - (optional) position of observer (in kpc; entries can be Quantity) 
           (default=[8.0,0.,0.]) OR Orbit object that corresponds to the orbit of the observer
(default=Object-wide default; can be Quantity)
           Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

        OUTPUT:

           ra(t) in deg

        HISTORY:

           2011-02-23 - Written - Bovy (NYU)

        """
        out= self._orb.ra(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def dec(self,*args,**kwargs):
        """
        NAME:

           dec

        PURPOSE:

           return the declination

        INPUT:

           t - (optional) time at which to get dec (can be Quantity)

           obs=[X,Y,Z] - (optional) position of observer (in kpc; entries can be Quantity) 
           (default=[8.0,0.,0.]) OR Orbit object that corresponds to the orbit of the observer
           Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

        OUTPUT:

           dec(t) in deg

        HISTORY:

           2011-02-23 - Written - Bovy (NYU)

        """
        out= self._orb.dec(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out
    
    def ll(self,*args,**kwargs):
        """
        NAME:

           ll

        PURPOSE:

           return Galactic longitude

        INPUT:

           t - (optional) time at which to get ll (can be Quantity)

           obs=[X,Y,Z] - (optional) position of observer (in kpc; entries can be Quantity) 
           (default=[8.0,0.,0.]) OR Orbit object that corresponds to the orbit of the observer
           Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

        OUTPUT:

           l(t) in deg

        HISTORY:

           2011-02-23 - Written - Bovy (NYU)

        """
        out= self._orb.ll(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def bb(self,*args,**kwargs):
        """
        NAME:

           bb

        PURPOSE:

           return Galactic latitude

        INPUT:

           t - (optional) time at which to get bb (can be Quantity)

           obs=[X,Y,Z] - (optional) position of observer (in kpc; entries can be Quantity) 
           (default=[8.0,0.,0.]) OR Orbit object that corresponds to the orbit of the observer
           Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

        OUTPUT:

           b(t) in deg

        HISTORY:

           2011-02-23 - Written - Bovy (NYU)

        """
        out= self._orb.bb(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def dist(self,*args,**kwargs):
        """
        NAME:

           dist

        PURPOSE:

           return distance from the observer

        INPUT:

           t - (optional) time at which to get dist (can be Quantity)

           obs=[X,Y,Z] - (optional) position of observer (in kpc; entries can be Quantity) 
           (default=[8.0,0.,0.]) OR Orbit object that corresponds to the orbit of the observer
           Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

        OUTPUT:

           dist(t) in kpc

        HISTORY:

           2011-02-23 - Written - Bovy (NYU)

        """
        out= self._orb.dist(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def pmra(self,*args,**kwargs):
        """
        NAME:

           pmra

        PURPOSE:

           return proper motion in right ascension (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmra (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantities)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           pm_ra(t) in mas/yr

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.pmra(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def pmdec(self,*args,**kwargs):
        """
        NAME:

           pmdec

        PURPOSE:

           return proper motion in declination (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmdec (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantities)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           pm_dec(t) in mas/yr

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.pmdec(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def pmll(self,*args,**kwargs):
        """
        NAME:

           pmll

        PURPOSE:

           return proper motion in Galactic longitude (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmll (can be Quantity)

v           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantities)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           pm_l(t) in mas/yr

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.pmll(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def pmbb(self,*args,**kwargs):
        """
        NAME:

           pmbb

        PURPOSE:

           return proper motion in Galactic latitude (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmbb (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           pm_b(t) in mas/yr

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.pmbb(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def vlos(self,*args,**kwargs):
        """
        NAME:

           vlos

        PURPOSE:

           return the line-of-sight velocity (in km/s)

        INPUT:

           t - (optional) time at which to get vlos (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           
           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           vlos(t) in km/s

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.vlos(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

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

           v_ra(t) in km/s

        HISTORY:

           2011-03-27 - Written - Bovy (NYU)

        """
        from .OrbitTop import _check_roSet, _check_voSet
        _check_roSet(self,kwargs,'vra')
        _check_voSet(self,kwargs,'vra')
        dist= self._orb.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            out= units.Quantity(dist.to(units.kpc).value*_K*
                                self._orb.pmra(*args,**kwargs)\
                                    .to(units.mas/units.yr).value,
                                unit=units.km/units.s)
        else:
            out= dist*_K*self._orb.pmra(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

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

           v_dec(t) in km/s

        HISTORY:

           2011-03-27 - Written - Bovy (NYU)

        """
        from .OrbitTop import _check_roSet, _check_voSet
        _check_roSet(self,kwargs,'vdec')
        _check_voSet(self,kwargs,'vdec')
        dist= self._orb.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            out= units.Quantity(dist.to(units.kpc).value*_K*
                                self._orb.pmdec(*args,**kwargs)\
                                    .to(units.mas/units.yr).value,
                                unit=units.km/units.s)
        else:
            out= dist*_K*self._orb.pmdec(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

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

           v_l(t) in km/s

        HISTORY:

           2011-03-27 - Written - Bovy (NYU)

        """
        from .OrbitTop import _check_roSet, _check_voSet
        _check_roSet(self,kwargs,'vll')
        _check_voSet(self,kwargs,'vll')
        dist= self._orb.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            out= units.Quantity(dist.to(units.kpc).value*_K*
                                self._orb.pmll(*args,**kwargs)\
                                    .to(units.mas/units.yr).value,
                                unit=units.km/units.s)
        else:
            out= dist*_K*self._orb.pmll(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out
        
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

           v_b(t) in km/s

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        from .OrbitTop import _check_roSet, _check_voSet
        _check_roSet(self,kwargs,'vbb')
        _check_voSet(self,kwargs,'vbb')
        dist= self._orb.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            out= units.Quantity(dist.to(units.kpc).value*_K*
                                self._orb.pmbb(*args,**kwargs)\
                                    .to(units.mas/units.yr).value,
                                unit=units.km/units.s)
        else:
            out= dist*_K*self._orb.pmbb(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def helioX(self,*args,**kwargs):
        """
        NAME:

           helioX

        PURPOSE:

           return Heliocentric Galactic rectangular x-coordinate (aka "X")

        INPUT:

           t - (optional) time at which to get X (can be Quantity)

           obs=[X,Y,Z] - (optional) position of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

        OUTPUT:

           helioX(t) in kpc

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.helioX(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def helioY(self,*args,**kwargs):
        """
        NAME:

           helioY

        PURPOSE:

           return Heliocentric Galactic rectangular y-coordinate (aka "Y")

        INPUT:

           t - (optional) time at which to get Y (can be Quantity)

           obs=[X,Y,Z] - (optional) position and of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.]; entries can be Quantity))
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

        OUTPUT:

           helioY(t) in kpc

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.helioY(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def helioZ(self,*args,**kwargs):
        """
        NAME:

           helioZ

        PURPOSE:

           return Heliocentric Galactic rectangular z-coordinate (aka "Z")

        INPUT:

           t - (optional) time at which to get Z (can be Quantity)

           obs=[X,Y,Z] - (optional) position of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

        OUTPUT:

           helioZ(t) in kpc

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.helioZ(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def U(self,*args,**kwargs):
        """
        NAME:

           U

        PURPOSE:

           return Heliocentric Galactic rectangular x-velocity (aka "U")

        INPUT:

           t - (optional) time at which to get U (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           U(t) in km/s

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.U(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def V(self,*args,**kwargs):
        """
        NAME:

           V

        PURPOSE:

           return Heliocentric Galactic rectangular y-velocity (aka "V")

        INPUT:

           t - (optional) time at which to get V (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           V(t) in km/s

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.V(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def W(self,*args,**kwargs):
        """
        NAME:

           W

        PURPOSE:

           return Heliocentric Galactic rectangular z-velocity (aka "W")

        INPUT:

           t - (optional) time at which to get W (can be Quantity)

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         in the Galactocentric frame
                         (in kpc and km/s) (default=[8.0,0.,0.,0.,220.,0.]; entries can be Quantity)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

        OUTPUT:

           W(t) in km/s

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        out= self._orb.W(*args,**kwargs)
        if len(out) == 1: return out[0]
        else: return out

    def SkyCoord(self,*args,**kwargs):
        """
        NAME:

           SkyCoord

        PURPOSE:

           return the position as an astropy SkyCoord

        INPUT:

           t - (optional) time at which to get the position (can be Quantity)

           obs=[X,Y,Z] - (optional) position of observer (in kpc; entries can be Quantity) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

        OUTPUT:

           SkyCoord(t)

        HISTORY:

           2015-06-02 - Written - Bovy (IAS)

        """
        return self._orb.SkyCoord(*args,**kwargs)

    def __call__(self,*args,**kwargs):
        """
        NAME:
 
          __call__

        PURPOSE:

           return the orbit at time t

        INPUT:

           t - desired time (can be Quantity)

           rect - if true, return rectangular coordinates

        OUTPUT:

           an Orbit instance with initial condition set to the 
           phase-space at time t or list of Orbit instances if multiple 
           times are given

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        orbSetupKwargs= {'ro':None,
                         'vo':None,
                         'zo':self._orb._zo,
                         'solarmotion':self._orb._solarmotion}
        if self._orb._roSet:
            orbSetupKwargs['ro']= self._orb._ro
        if self._orb._voSet:
            orbSetupKwargs['vo']= self._orb._vo
        thiso= self._orb(*args,**kwargs)
        if len(thiso.shape) == 1: return Orbit(vxvv=thiso,**orbSetupKwargs)
        else: return [Orbit(vxvv=thiso[:,ii],
                            **orbSetupKwargs) for ii in range(thiso.shape[1])]

    def plot(self,*args,**kwargs):
        """
        NAME:

           plot

        PURPOSE:

           plot a previously calculated orbit (with reasonable defaults)

        INPUT:

           d1= first dimension to plot ('x', 'y', 'R', 'vR', 'vT', 'z', 'vz', ...); can also be a user-defined function of time (e.g., lambda t: o.R(t) for R)

           d2= second dimension to plot; can also be a user-defined function of time (e.g., lambda t: o.R(t) for R)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           matplotlib.plot inputs+bovy_plot.plot inputs

        OUTPUT:

           sends plot to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._orb.plot(*args,**kwargs)

    def plot3d(self,*args,**kwargs):
        """
        NAME:

           plot3d

        PURPOSE:

           plot 3D aspects of an Orbit

        INPUT:

           d1= first dimension to plot ('x', 'y', 'R', 'vR', 'vT', 'z', 'vz', ...); can also be a user-defined function of time (e.g., lambda t: o.R(t) for R)

           d2= second dimension to plot

           d3= third dimension to plot

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           bovy_plot3d args and kwargs

        OUTPUT:

           plot

        HISTORY:

           2010-07-26 - Written - Bovy (NYU)

           2010-09-22 - Adapted to more general framework - Bovy (NYU)

           2010-01-08 - Adapted to 3D - Bovy (NYU)
        """
        return self._orb.plot3d(*args,**kwargs)

    def plotE(self,*args,**kwargs):
        """
        NAME:

           plotE

        PURPOSE:

           plot E(.) along the orbit

        INPUT:

           pot= Potential instance or list of instances in which the orbit was integrated

           d1= plot Ez vs d1: e.g., 't', 'z', 'R', 'vR', 'vT', 'vz'      

           normed= if set, plot E(t)/E(0) rather than E(t)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           +bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

           2014-06-16 - Changed to actually plot E rather than E/E0 - Bovy (IAS)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        return self._orb.plotE(*args,**kwargs)

    def plotEz(self,*args,**kwargs):
        """
        NAME:

           plotEz

        PURPOSE:

           plot E_z(.) along the orbit

        INPUT:

           pot=  Potential instance or list of instances in which the orbit was integrated

           d1= plot Ez vs d1: e.g., 't', 'z', 'R'

           normed= if set, plot E(t)/E(0) rather than E(t)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           +bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        return self._orb.plotEz(*args,**kwargs)

    def plotER(self,*args,**kwargs):
        """
        NAME:

           plotER

        PURPOSE:

           plot E_R(.) along the orbit

        INPUT:

           pot=  Potential instance or list of instances in which the orbit was integrated

           d1= plot ER vs d1: e.g., 't', 'z', 'R'

           normed= if set, plot E(t)/E(0) rather than E(t)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           +bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        return self._orb.plotER(*args,**kwargs)

    def plotEzJz(self,*args,**kwargs):
        """
        NAME:

           plotEzJzt

        PURPOSE:

           plot E_z(t)/sqrt(dens(R)) / (E_z(0)/sqrt(dens(R(0)))) along the orbit (an approximation to the vertical action)

        INPUT:

           pot - Potential instance or list of instances in which the orbit was integrated

           d1= plot Ez vs d1: e.g., 't', 'z', 'R'

           +bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-08-08 - Written - Bovy (NYU)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        return self._orb.plotEzJz(*args,**kwargs)

    def plotJacobi(self,*args,**kwargs):
        """
        NAME:

           plotJacobi

        PURPOSE:

           plot the Jacobi integral along the orbit

        INPUT:

           OmegaP= pattern speed

           pot= - Potential instance or list of instances in which the orbit 
                 was integrated

           d1= - plot Ez vs d1: e.g., 't', 'z', 'R', 'vR', 'vT', 'vz'      

           normed= if set, plot E(t)/E(0) rather than E(t)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           +bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2011-10-10 - Written - Bovy (IAS)

        """
        if not kwargs.get('pot',None) is None: kwargs['pot']= flatten_potential(kwargs.get('pot'))
        return self._orb.plotJacobi(*args,**kwargs)

    def plotR(self,*args,**kwargs):
        """
        NAME:

           plotR

        PURPOSE:

           plot R(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._orb.plotR(*args,**kwargs)

    def plotz(self,*args,**kwargs):
        """
        NAME:

           plotz

        PURPOSE:

           plot z(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._orb.plotz(*args,**kwargs)

    def plotvR(self,*args,**kwargs):
        """
        NAME:

           plotvR

        PURPOSE:

           plot vR(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._orb.plotvR(*args,**kwargs)

    def plotvT(self,*args,**kwargs):
        """
        NAME:

           plotvT

        PURPOSE:

           plot vT(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._orb.plotvT(*args,**kwargs)

    def plotphi(self,*args,**kwargs):
        """
        NAME:

           plotphi

        PURPOSE:

           plot \phi(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._orb.plotphi(*args,**kwargs)

    def plotvz(self,*args,**kwargs):
        """
        NAME:

           plotvz

        PURPOSE:

           plot vz(.) along the orbit

        INPUT:
           d1= plot vs d1: e.g., 't', 'z', 'R'

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._orb.plotvz(*args,**kwargs)

    def plotx(self,*args,**kwargs):
        """
        NAME:

           plotx

        PURPOSE:

           plot x(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

        """
        return self._orb.plotx(*args,**kwargs)

    def plotvx(self,*args,**kwargs):
        """
        NAME:

           plotvx

        PURPOSE:

           plot vx(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

        """
        return self._orb.plotvx(*args,**kwargs)

    def ploty(self,*args,**kwargs):
        """
        NAME:

           ploty

        PURPOSE:

           plot y(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

        """
        return self._orb.ploty(*args,**kwargs)

    def plotvy(self,*args,**kwargs):
        """
        NAME:

           plotvy

        PURPOSE:

           plot vy(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

        """
        return self._orb.plotvy(*args,**kwargs)

    def toPlanar(self):
        """
        NAME:

           toPlanar

        PURPOSE:

           convert a 3D orbit into a 2D orbit

        INPUT:

           (none)

        OUTPUT:

           planar Orbit

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

        """
        orbSetupKwargs= {'ro':None,
                         'vo':None,
                         'zo':self._orb._zo,
                         'solarmotion':self._orb._solarmotion}
        if self._orb._roSet:
            orbSetupKwargs['ro']= self._orb._ro
        if self._orb._voSet:
            orbSetupKwargs['vo']= self._orb._vo
        if len(self._orb.vxvv) == 6:
            vxvv= [self._orb.vxvv[0],self._orb.vxvv[1],
                   self._orb.vxvv[2],self._orb.vxvv[5]]
        elif len(self._orb.vxvv) == 5:
            vxvv= [self._orb.vxvv[0],self._orb.vxvv[1],self._orb.vxvv[2]]
        else:
            raise AttributeError("planar or linear Orbits do not have the toPlanar attribute")
        return Orbit(vxvv=vxvv,**orbSetupKwargs)

    def toLinear(self):
        """
        NAME:

           toLinear

        PURPOSE:

           convert a 3D orbit into a 1D orbit (z)

        INPUT:

           (none)

        OUTPUT:

           linear Orbit

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

        """
        orbSetupKwargs= {'ro':None,
                         'vo':None,
                         'zo':self._orb._zo,
                         'solarmotion':self._orb._solarmotion}
        if self._orb._roSet:
            orbSetupKwargs['ro']= self._orb._ro
        if self._orb._voSet:
            orbSetupKwargs['vo']= self._orb._vo
        if len(self._orb.vxvv) == 6 or len(self._orb.vxvv) == 5:
            vxvv= [self._orb.vxvv[3],self._orb.vxvv[4]]
        else:
            raise AttributeError("planar or linear Orbits do not have the toPlanar attribute")
        return Orbit(vxvv=vxvv,**orbSetupKwargs)

    def __add__(self,linOrb):
        """
        NAME:

           __add__

        PURPOSE:

           add a linear orbit and a planar orbit to make a 3D orbit

        INPUT:

           linear or plane orbit instance

        OUTPUT:

           3D orbit

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

        """
        orbSetupKwargs= {'ro':None,
                         'vo':None,
                         'zo':self._orb._zo,
                         'solarmotion':self._orb._solarmotion}
        if self._orb._roSet:
            orbSetupKwargs['ro']= self._orb._ro
        if self._orb._voSet:
            orbSetupKwargs['vo']= self._orb._vo
        if (not (isinstance(self._orb,planarOrbitTop) and 
                isinstance(linOrb._orb,linearOrbit)) and
            not (isinstance(self._orb,linearOrbit) and 
                 isinstance(linOrb._orb,planarOrbitTop))):
            raise AttributeError("Only planarOrbit+linearOrbit is supported")
        if isinstance(self._orb,planarROrbit):
            return Orbit(vxvv=[self._orb.vxvv[0],self._orb.vxvv[1],
                               self._orb.vxvv[2],
                               linOrb._orb.vxvv[0],linOrb._orb.vxvv[1]],
                         **orbSetupKwargs)
        elif isinstance(self._orb,planarOrbit):
            return Orbit(vxvv=[self._orb.vxvv[0],self._orb.vxvv[1],
                               self._orb.vxvv[2],
                               linOrb._orb.vxvv[0],linOrb._orb.vxvv[1],
                               self._orb.vxvv[3]],
                         **orbSetupKwargs)
        elif isinstance(linOrb._orb,planarROrbit):
            return Orbit(vxvv=[linOrb._orb.vxvv[0],linOrb._orb.vxvv[1],
                               linOrb._orb.vxvv[2],
                               self._orb.vxvv[0],self._orb.vxvv[1]],
                         **orbSetupKwargs)
        elif isinstance(linOrb._orb,planarOrbit):
            return Orbit(vxvv=[linOrb._orb.vxvv[0],linOrb._orb.vxvv[1],
                               linOrb._orb.vxvv[2],
                               self._orb.vxvv[0],self._orb.vxvv[1],
                               linOrb._orb.vxvv[3]],
                         **orbSetupKwargs)

    def animate(self,*args,**kwargs): #pragma: no cover
        """
        NAME:

           animate

        PURPOSE:

           animate a previously calculated orbit (with reasonable defaults)

        INPUT:

           d1= first dimension to plot ('x', 'y', 'R', 'vR', 'vT', 'z', 'vz', ...); can be list with up to three entries for three subplots; each entry can also be a user-defined function of time (e.g., lambda t: o.R(t) for R)

           d2= second dimension to plot; can be list with up to three entries for three subplots; each entry can also be a user-defined function of time (e.g., lambda t: o.R(t) for R)

           width= (600) width of output div in px

           height= (400) height of output div in px

           xlabel= (pre-defined labels) label for the first dimension (or list of labels if d1 is a list); should only have to be specified when using a function as d1 and can then specify as, e.g., [None,'YOUR LABEL',None] if d1 is a list of three xs and the first and last are standard entries)

           ylabel= (pre-defined labels) label for the second dimension (or list of labels if d2 is a list); should only have to be specified when using a function as d2 and can then specify as, e.g., [None,'YOUR LABEL',None] if d1 is a list of three xs and the first and last are standard entries)

           json_filename= (None) if set, save the data necessary for the figure in this filename (e.g.,  json_filename= 'orbit_data/orbit.json'); this path is also used in the output HTML, so needs to be accessible

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           IPython.display.HTML object with code to animate the orbit; can be directly shown in jupyter notebook or embedded in HTML pages; get a text version of the HTML using the _repr_html_() function

        HISTORY:

           2017-09-17-24 - Written - Bovy (UofT)

        """
        return self._orb.animate(*args,**kwargs)

    @classmethod
    def from_name(cls, name, vo=None, ro=None, zo=None, solarmotion=None):
        """
        NAME:

            from_name

        PURPOSE:

            given the name of an object, retrieve coordinate information for that object from SIMBAD and return a corresponding orbit

        INPUT:

            name - the name of the object

            +standard Orbit initialization keywords:

                ro= distance from vantage point to GC (kpc; can be Quantity)

                vo= circular velocity at ro (km/s; can be Quantity)

                zo= offset toward the NGP of the Sun wrt the plane (kpc; can be Quantity; default = 25 pc)

                solarmotion= 'hogg' or 'dehnen', or 'schoenrich', or value in [-U,V,W]; can be Quantity

        OUTPUT:

            orbit containing the phase space coordinates of the named object

        HISTORY:

            2018-07-15 - Written - Mathew Bub (UofT)

        """
        if not _APY_LOADED: # pragma: no cover
            raise ImportError('astropy needs to be installed to use '
                              'Orbit.from_name')
        if not _ASTROQUERY_LOADED: # pragma: no cover
            raise ImportError('astroquery needs to be installed to use '
                              'Orbit.from_name')

        # setup a SIMBAD query with the appropriate fields
        simbad= Simbad()
        simbad.add_votable_fields('ra(d)', 'dec(d)', 'pmra', 'pmdec',
                                  'rv_value', 'plx', 'distance')
        simbad.remove_votable_fields('main_id', 'coordinates')

        # query SIMBAD for the named object
        try:
            simbad_table= simbad.query_object(name)
        except OSError: # pragma: no cover
            raise OSError('failed to connect to SIMBAD')
        if not simbad_table:
            raise ValueError('failed to find {} in SIMBAD'.format(name))

        # check that the necessary coordinates have been found
        missing= simbad_table.mask
        if (any(missing['RA_d', 'DEC_d', 'PMRA', 'PMDEC', 'RV_VALUE'][0]) or
                all(missing['PLX_VALUE', 'Distance_distance'][0])):
            raise ValueError('failed to find some coordinates for {} in '
                             'SIMBAD'.format(name))
        ra, dec, pmra, pmdec, vlos= simbad_table['RA_d', 'DEC_d', 'PMRA',
                                                 'PMDEC', 'RV_VALUE'][0]

        # get a distance value
        if not missing['PLX_VALUE'][0]:
            dist= 1/simbad_table['PLX_VALUE'][0]
        else:
            dist_str= str(simbad_table['Distance_distance'][0]) + \
                      simbad_table['Distance_unit'][0]
            dist= units.Quantity(dist_str).to(units.kpc).value

        return cls(vxvv=[ra,dec,dist,pmra,pmdec,vlos], radec=True, ro=ro, vo=vo,
                   zo=zo, solarmotion=solarmotion)

def _check_integrate_dt(t,dt):
    """Check that the stepszie in t is an integer x dt"""
    if dt is None:
        return True
    mult= round((t[1]-t[0])/dt)
    if nu.fabs(mult*dt-t[1]+t[0]) < 10.**-10.:
        return True
    else:
        return False

def _check_potential_dim(orb,pot):
    from galpy.potential import _dim
    # Don't deal with pot=None here, just dimensionality
    assert pot is None or orb.dim() <= _dim(pot), 'Orbit dimensionality is %i, but potential dimensionality is %i < %i; orbit needs to be of equal or lower dimensionality as the potential; you can reduce the dimensionality---if appropriate---of your orbit with orbit.toPlanar or orbit.toLinear' % (orb.dim(),_dim(pot),orb.dim())

def _check_consistent_units(orb,pot):
    if pot is None: return None
    if isinstance(pot,list):
        if orb._roSet and pot[0]._roSet:
            assert nu.fabs(orb._ro-pot[0]._ro) < 10.**-10., 'Physical conversion for the Orbit object is not consistent with that of the Potential given to it'
        if orb._voSet and pot[0]._voSet:
            assert nu.fabs(orb._vo-pot[0]._vo) < 10.**-10., 'Physical conversion for the Orbit object is not consistent with that of the Potential given to it'
    else:
        if orb._roSet and pot._roSet:
            assert nu.fabs(orb._ro-pot._ro) < 10.**-10., 'Physical conversion for the Orbit object is not consistent with that of the Potential given to it'
        if orb._voSet and pot._voSet:
            assert nu.fabs(orb._vo-pot._vo) < 10.**-10., 'Physical conversion for the Orbit object is not consistent with that of the Potential given to it'
    return None
