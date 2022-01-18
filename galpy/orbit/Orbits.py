from six import raise_from
from past.builtins import basestring
import os
import sys
_PY3= sys.version > '3'
import json
import copy
import string
from functools import wraps
from random import choice
from string import ascii_lowercase
from pkg_resources import parse_version
import warnings
import numpy
from scipy import interpolate, optimize
import scipy
_SCIPY_VERSION= parse_version(scipy.__version__)
if _SCIPY_VERSION < parse_version('0.10'): #pragma: no cover
    from scipy.maxentropy import logsumexp
elif _SCIPY_VERSION < parse_version('0.19'): #pragma: no cover
    from scipy.misc import logsumexp
else:
    from scipy.special import logsumexp
from ..util import galpyWarning, galpyWarningVerbose
from ..util.conversion import physical_conversion, physical_compatible
from ..util.coords import _K
from ..util import coords
from ..util import plot
from ..util import conversion
from ..potential import toPlanarPotential, PotentialError, evaluatePotentials,\
    evaluateplanarPotentials, evaluatelinearPotentials
from ..potential import flatten as flatten_potential
from ..potential.Potential import _check_c
from ..potential import rl, _isNonAxi
from ..potential.DissipativeForce import _isDissipative
from .integrateLinearOrbit import integrateLinearOrbit_c, _ext_loaded, \
    integrateLinearOrbit
from .integratePlanarOrbit import integratePlanarOrbit_c, \
    integratePlanarOrbit, integratePlanarOrbit_dxdv
from .integrateFullOrbit import integrateFullOrbit_c, integrateFullOrbit
ext_loaded= _ext_loaded
_APY_LOADED= True
try:
    from astropy.coordinates import SkyCoord
except ImportError:
    SkyCoord = None
    _APY_LOADED= False
if _APY_LOADED:
    from astropy import units, coordinates
    import astropy
    _APY3= parse_version(astropy.__version__) > parse_version('3')
    _APY_GE_31= parse_version(astropy.__version__) > parse_version('3.0.5')
_ASTROQUERY_LOADED= True
try:
    from astroquery.simbad import Simbad
except ImportError:
    _ASTROQUERY_LOADED= False
from ..util import config
_APY_UNITS= config.__config__.getboolean('astropy','astropy-units')
if _APY_LOADED:
    vxvv_units= [units.kpc,units.km/units.s,units.km/units.s,
                 units.kpc,units.km/units.s,units.rad]
else:
    _APY_UNITS= False
# Set default numcores for integrate w/ parallel map using OMP_NUM_THREADS
try:
    _NUMCORES= int(os.environ['OMP_NUM_THREADS'])
except KeyError:
    import multiprocessing
    _NUMCORES= multiprocessing.cpu_count()
# named_objects file
def _named_objects_key_formatting(name):
    # Remove punctuation, spaces, and make lowercase
    if _PY3:
        out_name= name.translate(\
            str.maketrans('', '',string.punctuation)).replace(' ', '').lower()
    else: #pragma: no cover
        out_name= str(name).translate(None,string.punctuation)\
            .replace(' ', '').lower()
    return out_name   
_known_objects= None
_known_objects_original_keys= None # these are use for auto-completion
_known_objects_collections_original_keys= None
_known_objects_synonyms_original_keys= None
_known_objects_keys_updated= False
def _load_named_objects():
    global _known_objects
    global _known_objects_original_keys
    global _known_objects_collections_original_keys
    global _known_objects_synonyms_original_keys
    if not _known_objects:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'named_objects.json'),'r') as jsonfile:
            _known_objects= json.load(jsonfile)
        # Save original keys for auto-completion
        _known_objects_original_keys= copy.copy(list(_known_objects.keys()))
        _known_objects_collections_original_keys= \
            copy.copy(list(_known_objects['_collections'].keys()))
        _known_objects_synonyms_original_keys= \
            copy.copy(list(_known_objects['_synonyms'].keys()))
        # Add synonyms as duplicates
        for name in _known_objects['_synonyms']:
            _known_objects[name]= \
                _known_objects[_known_objects['_synonyms'][name]]
    return None
def _update_keys_named_objects():
    global _known_objects_keys_updated
    if not _known_objects_keys_updated:
        # Format the keys of the known objects dictionary, first collections
        old_keys= list(_known_objects['_collections'].keys())
        for old_key in old_keys:
            _known_objects['_collections']\
                [_named_objects_key_formatting(old_key)]= \
                   _known_objects['_collections'].pop(old_key)
        # Then the objects themselves
        old_keys= list(_known_objects.keys())
        old_keys.remove('_collections')
        old_keys.remove('_synonyms')
        for old_key in old_keys:
            _known_objects[_named_objects_key_formatting(old_key)]= \
                   _known_objects.pop(old_key)
        _known_objects_keys_updated= True
# Auto-completion
try: # pragma: no cover
    from IPython import get_ipython
    _load_named_objects()
    def name_completer(ipython,event):
        try: # encapsulate in try/except to avoid *any* error
            out= copy.copy(_known_objects_original_keys)
            out.remove('_collections')
            out.remove('_synonyms')
            out.extend(_known_objects_collections_original_keys)
            out.extend(_known_objects_synonyms_original_keys)
            out.extend(['ro=','vo=','zo=','solarmotion='])
        except: pass
        return out
    get_ipython().set_hook('complete_command',name_completer,
                           re_key=".*from_name")
except: pass

def shapeDecorator(func):
    """Decorator to return Orbits outputs with the correct shape"""
    @wraps(func)
    def shape_wrapper(*args,**kwargs):
        dontreshape= kwargs.get('dontreshape',False)
        result= func(*args,**kwargs)
        if dontreshape:
            return result
        elif args[0].shape == ():
            return result[0]
        else:
            return numpy.reshape(result,args[0].shape+result.shape[1:])
    return shape_wrapper
class Orbit(object):
    """
    Class representing single and multiple orbits.
    """
    def __init__(self,vxvv=None,ro=None,vo=None,zo=None,solarmotion=None,
                 radec=False,uvw=False,lb=False):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize an Orbit instance

        INPUT:

            vxvv - initial conditions (must all have the same phase-space dimension); can be either

                a) astropy (>v3.0) SkyCoord with arbitrary shape, including velocities (note that this turns *on* physical output even if ro and vo are not given)

                b) array of arbitrary shape (shape,phasedim) (shape of the orbits, followed by the phase-space dimension of the orbit); shape information is retained and used in outputs; elements can be either

                    1) in Galactocentric cylindrical coordinates with phase-space coordinates arranged as [R,vR,vT(,z,vz,phi)]; needs to be in internal units (for Quantity input; see 'list' option below)

                    2) [ra,dec,d,mu_ra, mu_dec,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (ICRS; mu_ra = mu_ra * cos dec); (for Quantity input, see 'list' option below);

                    4) [ra,dec,d,U,V,W] in [deg,deg,kpc,km/s,km/s,kms]; (for Quantity input; see 'list' option below); ICRS frame

                    5) (l,b,d,mu_l, mu_b, vlos) in [deg,deg,kpc,mas/yr,mas/yr,km/s) (mu_l = mu_l * cos b); (for Quantity input; see 'list' option below)

                    6) [l,b,d,U,V,W] in [deg,deg,kpc,km/s,km/s,kms]; (for Quantity input; see 'list' option below)

                    5) and 6) also work when leaving out b and mu_b/W

                c) lists of initial conditions, entries can be

                    1) individual Orbit instances (of single objects)

                    2) Quantity arrays arranged as in section 2) above (so things like [R,vR,vT,z,vz,phi], where R, vR, ... can be arbitrary shape Quantity arrays)

                    3) list of Quantities (so things like [R1,vR1,..,], where R1, vR1, ... are scalar Quantities

                    4) None: assumed to be the Sun; if None occurs in a list it is assumed to be the Sun *and all other items in the list are assumed to be [ra,dec,...]*; cannot be combined with Quantity lists (2 and 3 above)

                    5) lists of scalar phase-space coordinates arranged as in b) (so things like [R,vR,...] where R,vR are scalars in internal units

        OPTIONAL INPUTS:

            ro - distance from vantage point to GC (kpc; can be Quantity)

            vo - circular velocity at ro (km/s; can be Quantity)

            zo - offset toward the NGP of the Sun wrt the plane (kpc; can be Quantity; default = 20.8 pc from Bennett & Bovy 2019)

            solarmotion - 'hogg' or 'dehnen', or 'schoenrich', or value in [-U,V,W]; can be Quantity

        OUTPUT:

            instance

        HISTORY:

            2018-10-13 - Written - Mathew Bub (UofT)

            2019-01-01 - Better handling of unit/coordinate-conversion parameters and consistency checks - Bovy (UofT)

            2019-02-01 - Handle array of SkyCoords in a faster way by making use of the fact that array of SkyCoords is processed correctly by Orbit

            2019-02-18 - Don't support radec, lb, or uvw keywords to avoid slow coordinate transformations that would require ugly code to fix - Bovy (UofT)

            2019-03-19 - Allow array vxvv and arbitrary shapes - Bovy (UofT)

        """
        # First deal with None = Sun
        if vxvv is None: # Assume one wants the Sun
            vxvv= numpy.array([0.,0.,0.,0.,0.,0.])
            radec= True
        elif isinstance(vxvv,(list, tuple)):
            if None in vxvv:
                vxvv= [[0.,0.,0.,0.,0.,0.] 
                       if tvxvv is None else tvxvv
                       for tvxvv in vxvv]
                radec= True
        # Set ro, vo, zo, solarmotion based on input, SkyCoord vxvv, ...
        self._setup_parse_coordtransform(vxvv,ro,vo,zo,solarmotion,
                                         radec,lb)
        # Determine and record input shape and flatten for further processing
        if _APY_LOADED and isinstance(vxvv,SkyCoord):
            input_shape= vxvv.shape
            vxvv=vxvv.flatten()
        elif isinstance(vxvv,numpy.ndarray):
            input_shape= vxvv.shape[:-1]
            vxvv= numpy.atleast_2d(vxvv)
            vxvv= vxvv.reshape((numpy.prod(vxvv.shape[:-1]),vxvv.shape[-1]))
        elif isinstance(vxvv,(list, tuple)):
            if isinstance(vxvv[0],Orbit):
                vxvv= self._setup_parse_listofOrbits(vxvv,ro,vo,zo,solarmotion)
                input_shape= (len(vxvv),)
                vxvv= numpy.array(vxvv)
            elif _APY_LOADED and isinstance(vxvv[0],units.Quantity):
                # Case where vxvv= [R,vR,...] or [ra,dec,...] with Quantities
                input_shape= vxvv[0].shape
                vxvv= [s.flatten() for s in vxvv]
                # Keep as list, is fine later...
            elif _APY_LOADED and isinstance(vxvv[0],list) \
                    and isinstance(vxvv[0][0],units.Quantity):
                # Case where vxvv= [[R1,vR1,...],[R2,vR2,...]]
                # or [[ra1,dec1,...],[ra2,dec2,...]] with Quantities
                input_shape= (len(vxvv),)
                pdim= len(vxvv[0])
                stack= []
                for pp in range(pdim):
                    stack.append(\
                        numpy.array([tvxvv[pp].to(vxvv[0][pp].unit).value
                                     for tvxvv in vxvv])\
                            *vxvv[0][pp].unit)
                vxvv= stack
                # Keep as list, is fine later...
            elif numpy.ndim(vxvv[0]) == 0: # Scalar, so assume single object
                vxvv= [vxvv]
                input_shape= ()
                vxvv= numpy.array(vxvv)
            else:
                input_shape= (len(vxvv),)
                vxvv= numpy.array(vxvv)
            if isinstance(vxvv,numpy.ndarray) and vxvv.dtype == 'object':
                # if diff. phasedim, object array is created
                raise RuntimeError("All individual orbits in an Orbit class must have the same phase-space dimensionality")
        #: Tuple of Orbit dimensions
        self.shape= input_shape
        self._setup_parse_vxvv(vxvv,radec,lb,uvw)
        # Check that we have a valid phase-space dim (often messed up by not 
        # transposing the input array to the correct shape)
        if self.phasedim() < 2 or self.phasedim() > 6:
            if len(self.vxvv) > 1 and len(self.vxvv) < 7:
                raise RuntimeError("Invalid phase-space dimension {:d} for {:d} objects; perhaps you meant to transpose the input?".format(self.phasedim(),len(self.vxvv)))
            else:
                raise RuntimeError("Invalid phase-space dimension: phasedim = {:d}, but should be between 2 and 6".format(self.phasedim()))
        #: Total number of elements in the Orbit instance
        self.size= 1 if self.shape == () else len(self.vxvv)
        if self.dim() == 1:
            # For the 1D case, solar position/velocity is not used currently
            self._zo= None
            self._solarmotion= None

    def _setup_parse_coordtransform(self,vxvv,ro,vo,zo,solarmotion,
                                    radec,lb):
        # Parse coordinate-transformation inputs with units
        ro= conversion.parse_length_kpc(ro)
        zo= conversion.parse_length_kpc(zo)
        vo= conversion.parse_velocity_kms(vo)
        # if vxvv is SkyCoord, preferentially use its ro and zo
        if _APY_LOADED and isinstance(vxvv,SkyCoord):
            if not _APY3: # pragma: no cover
                raise ImportError('Orbit initialization using an astropy SkyCoord requires astropy >3.0')
            if zo is None and not vxvv.z_sun is None:
                zo= vxvv.z_sun.to(units.kpc).value
            elif not vxvv.z_sun is None:
                if numpy.fabs(zo-vxvv.z_sun.to(units.kpc).value) > 1e-8:
                    raise ValueError("Orbit initialization's zo different from SkyCoord's z_sun; these should be the same for consistency")
            elif zo is None and not vxvv.galcen_distance is None:
                zo= 0.
            if ro is None and not vxvv.galcen_distance is None:
                ro= numpy.sqrt(vxvv.galcen_distance.to(units.kpc).value**2.
                            -zo**2.)
            elif not vxvv.galcen_distance is None and \
                    numpy.fabs(ro**2.+zo**2.-vxvv.galcen_distance.to(units.kpc).value**2.) > 1e-10:
                warnings.warn("Orbit's initialization normalization ro and zo are incompatible with SkyCoord's galcen_distance (should have galcen_distance^2 = ro^2 + zo^2)",galpyWarning)
        # If at this point ro/vo not set, use default from config
        if (_APY_LOADED and isinstance(vxvv,SkyCoord)) or radec or lb:
            if ro is None:
                ro= config.__config__.getfloat('normalization','ro')
            if vo is None:
                vo= config.__config__.getfloat('normalization','vo')
        # If at this point zo not set, use default
        if zo is None: zo= 0.0208
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
            vsolar= numpy.array([-10.1,4.0,6.7])
        elif isinstance(solarmotion,str) and solarmotion.lower() == 'dehnen':
            vsolar= numpy.array([-10.,5.25,7.17])
        elif isinstance(solarmotion,str) \
                and solarmotion.lower() == 'schoenrich':
            vsolar= numpy.array([-11.1,12.24,7.25])
        else:
            vsolar= numpy.array(conversion.parse_velocity_kms(solarmotion))
        # If both vxvv SkyCoord with vsun and solarmotion set, check the same
        if _APY_LOADED and isinstance(vxvv,SkyCoord) \
                and not vxvv.galcen_v_sun is None:
            if numpy.any(numpy.fabs(sc_solarmotion-vsolar) > 1e-8):
                raise ValueError("Orbit initialization's solarmotion parameter not compatible with SkyCoord's galcen_v_sun; these should be the same for consistency (this may be because you did not set vo; galcen_v_sun = solarmotion+vo for consistency)")
        # Now store all coordinate-transformation parameters and save whether
        # ro/vo are set (they are considered to be set if they have been
        # determined at this point, even if they were not explicitly set
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
        self._zo= zo
        self._solarmotion= vsolar
        return None

    def _setup_parse_listofOrbits(self,vxvv,ro,vo,zo,solarmotion):
        # Only implement lists of scalar Orbit for now
        if not numpy.all([o.shape == () for o in vxvv]):
            raise RuntimeError("Initializing an Orbit instance with a list of Orbit instances only supports lists of single Orbit instances")
        # Need to check that coordinate-transformation parameters are 
        # consistent between given orbits and between this instance's
        # initialization and the given orbits; if not explicitly given
        # for this instance, fall back onto list's parameters
        ros= numpy.array([o._ro for o in vxvv])
        vos= numpy.array([o._vo for o in vxvv])
        zos= numpy.array([o._zo for o in vxvv])
        solarmotions= numpy.array([o._solarmotion for o in vxvv])
        if numpy.any(numpy.fabs(ros-ros[0]) > 1e-10):
            raise RuntimeError("All individual orbits given to an Orbit class must have the same ro unit-conversion parameter")
        if numpy.any(numpy.fabs(vos-vos[0]) > 1e-10):
            raise RuntimeError("All individual orbits given to an Orbit class must have the same vo unit-conversion parameter")
        if not zos[0] is None and numpy.any(numpy.fabs(zos-zos[0]) > 1e-10):
            raise RuntimeError("All individual orbits given to an Orbit class must have the same zo solar offset")               
        if not solarmotions[0] is None and \
                numpy.any(numpy.fabs(solarmotions-solarmotions[0]) > 1e-10):
            raise RuntimeError("All individual orbits given to an Orbit class must have the same solar motion")
        if self._roSet:
            if numpy.fabs(ros[0]-self._ro) > 1e-10:
                raise RuntimeError("All individual orbits given to an Orbit class must have the same ro unit-conversion parameter as used in the initialization call")
        else:
            self._ro= vxvv[0]._ro
            self._roSet= vxvv[0]._roSet
        if self._voSet:
            if numpy.fabs(vos[0]-self._vo) > 1e-10:
                raise RuntimeError("All individual orbits given to an Orbit class must have the same vo unit-conversion parameter as used in the initialization call")
        else:
            self._vo= vxvv[0]._vo
            self._voSet= vxvv[0]._voSet
        if not zo is None:
            if numpy.fabs(zos[0]-self._zo) > 1e-10:
                raise RuntimeError("All individual orbits given to an Orbit class must have the same zo solar offset parameter as used in the initialization call")
        else:
            self._zo= vxvv[0]._zo
        if not solarmotion is None:
            if numpy.any(numpy.fabs(solarmotions[0]-self._solarmotion) > 1e-10):
                raise RuntimeError("All individual orbits given to an Orbit class must have the same solar motion as used in the initialization call")
        else:
            self._solarmotion= vxvv[0]._solarmotion
        # shape of o.vxvv is (1,phasedim) due to internal storage
        return [list(o.vxvv[0]) for o in vxvv]
                
    def _setup_parse_vxvv(self,vxvv,radec,lb,uvw):
        if _APY_LOADED and isinstance(vxvv,SkyCoord):
            galcen_v_sun= coordinates.CartesianDifferential(\
                numpy.array([-self._solarmotion[0],
                              self._solarmotion[1]+self._vo,
                              self._solarmotion[2]])*units.km/units.s)
            gc_frame= coordinates.Galactocentric(\
                galcen_distance=numpy.sqrt(self._ro**2.+self._zo**2.)\
                                    *units.kpc,
                z_sun=self._zo*units.kpc,galcen_v_sun=galcen_v_sun)
            vxvvg= vxvv.transform_to(gc_frame)
            if _APY_GE_31:
                vxvvg.representation_type= 'cylindrical'
            else: #pragma: no cover
                vxvvg.representation= 'cylindrical'
            R= vxvvg.rho.to(units.kpc).value/self._ro
            phi= numpy.pi-vxvvg.phi.to(units.rad).value
            z= vxvvg.z.to(units.kpc).value/self._ro
            try:
                vR= vxvvg.d_rho.to(units.km/units.s).value/self._vo
            except TypeError:
                raise TypeError("SkyCoord given to Orbit initialization does not have velocity data, which is required to setup an Orbit")
            vT= -(vxvvg.d_phi*vxvvg.rho)\
                .to(units.km/units.s,
                    equivalencies=units.dimensionless_angles()).value/self._vo
            vz= vxvvg.d_z.to(units.km/units.s).value/self._vo
            vxvv= numpy.array([R,vR,vT,z,vz,phi])
            # Make sure radec and lb are False (issue #402)
            radec= False
            lb= False
        elif not isinstance(vxvv,(list, tuple)):
            vxvv= vxvv.T # (norb,phasedim) --> (phasedim,norb) easier later
        if not (_APY_LOADED and isinstance(vxvv,SkyCoord)) and (radec or lb):
            if radec:
                if _APY_LOADED and isinstance(vxvv[0],units.Quantity):
                    ra, dec= vxvv[0].to(units.deg).value, \
                        vxvv[1].to(units.deg).value
                else:
                    ra, dec= vxvv[0], vxvv[1]
                l,b= coords.radec_to_lb(ra,dec,degree=True,epoch=None).T
                _extra_rot= True
            elif len(vxvv) == 4:
                l, b= vxvv[0], numpy.zeros_like(vxvv[0])
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
                                             degree=True).T
                else:
                    X,Y,Z= coords.lbd_to_XYZ(l,b,vxvv[2],degree=True).T
                vx= conversion.parse_velocity_kms(vxvv[3])
                vy= conversion.parse_velocity_kms(vxvv[4])
                vz= conversion.parse_velocity_kms(vxvv[5])
            else:
                if radec:
                    if _APY_LOADED and isinstance(vxvv[3],units.Quantity):
                        pmra, pmdec= vxvv[3].to(units.mas/units.yr).value, \
                            vxvv[4].to(units.mas/units.yr).value
                    else:
                        pmra, pmdec= vxvv[3], vxvv[4]
                    pmll, pmbb= coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec,
                                                             degree=True,
                                                             epoch=None).T
                    d, vlos= vxvv[2], vxvv[5]
                elif len(vxvv) == 4:
                    pmll, pmbb= vxvv[2], numpy.zeros_like(vxvv[2])
                    d, vlos= vxvv[1], vxvv[3]
                else:
                    pmll, pmbb= vxvv[3], vxvv[4]
                    d, vlos= vxvv[2], vxvv[5]
                d= conversion.parse_length_kpc(d)
                vlos= conversion.parse_velocity_kms(vlos)
                if _APY_LOADED and isinstance(pmll,units.Quantity):
                    pmll= pmll.to(units.mas/units.yr).value
                if _APY_LOADED and isinstance(pmbb,units.Quantity):
                    pmbb= pmbb.to(units.mas/units.yr).value
                X,Y,Z,vx,vy,vz= coords.sphergal_to_rectgal(l,b,d,
                                                           vlos,pmll, pmbb,
                                                           degree=True).T
            X/= self._ro
            Y/= self._ro
            Z/= self._ro
            vx/= self._vo
            vy/= self._vo
            vz/= self._vo
            vsun= numpy.array([0.,1.,0.,])+self._solarmotion/self._vo
            R, phi, z= coords.XYZ_to_galcencyl(X,Y,Z,Zsun=self._zo/self._ro,
                                               _extra_rot=_extra_rot).T
            vR, vT,vz= coords.vxvyvz_to_galcencyl(vx,vy,vz,
                                                  R,phi,z,
                                                  vsun=vsun,
                                                  Xsun=1.,Zsun=self._zo/self._ro,
                                                  galcen=True,
                                                  _extra_rot=_extra_rot).T
            if lb and len(vxvv) == 4: vxvv= numpy.array([R,vR,vT,phi])
            else: vxvv= numpy.array([R,vR,vT,z,vz,phi])
        # Parse vxvv if it consists of Quantities
        if _APY_LOADED and isinstance(vxvv[0],units.Quantity):
            # Need to set ro and vo, default if not specified, so need to
            # turn them on
            self._roSet= True
            self._voSet= True
            new_vxvv= [vxvv[0].to(vxvv_units[0]).value/self._ro,
                       vxvv[1].to(vxvv_units[1]).value/self._vo]
            if len(vxvv) > 2:
                new_vxvv.append(vxvv[2].to(vxvv_units[2]).value/self._vo)
            if len(vxvv) == 4:
                new_vxvv.append(vxvv[3].to(vxvv_units[5]).value)
            elif len(vxvv) > 4:
                new_vxvv.append(vxvv[3].to(vxvv_units[3]).value/self._ro)
                new_vxvv.append(vxvv[4].to(vxvv_units[4]).value/self._vo)
                if len(vxvv) == 6:
                    new_vxvv.append(vxvv[5].to(vxvv_units[5]).value)
            vxvv= numpy.array(new_vxvv)
        # (phasedim,norb) --> (norb,phasedim) again and store
        self.vxvv= vxvv.T
        return None

    @classmethod
    def from_name(cls,*args,**kwargs):
        """
        NAME:

            from_name

        PURPOSE:

            given the name of an object or a list of names, retrieve coordinate information for that object from SIMBAD and return a corresponding orbit

        INPUT:

            name - the name of the object or list of names; when loading a collection of objects (like 'mwglobularclusters'), lists are not allowed

            +standard Orbit initialization keywords:

                ro= distance from vantage point to GC (kpc; can be Quantity)

                vo= circular velocity at ro (km/s; can be Quantity)

                zo= offset toward the NGP of the Sun wrt the plane (kpc; can be Quantity; default = 20.8 pc from Bennett & Bovy 2019)

                solarmotion= 'hogg' or 'dehnen', or 'schoenrich', or value in [-U,V,W]; can be Quantity

        OUTPUT:

            orbit containing the phase space coordinates of the named object

        HISTORY:

            2018-07-15 - Written - Mathew Bub (UofT)

            2019-05-21 - Generalized to multiple objects and incorporated into Orbits - Bovy (UofT)

        """
        if not _APY_LOADED: # pragma: no cover
            raise ImportError('astropy needs to be installed to use '
                              'Orbit.from_name')
        if not _ASTROQUERY_LOADED: # pragma: no cover
            raise ImportError('astroquery needs to be installed to use '
                              'Orbit.from_name')
        _load_named_objects()
        _update_keys_named_objects()
        # Stack coordinate-transform parameters, so they can be changed...
        obs= numpy.array([kwargs.get('ro',None),
                          kwargs.get('vo',None),
                          kwargs.get('zo',None),
                          kwargs.get('solarmotion',None)],
                         dtype='object')
        if len(args) > 1:
            name= [n for n in args]
        elif isinstance(args[0],list):
            name= args[0]
        else:
            this_name= _named_objects_key_formatting(args[0])
            if this_name in _known_objects['_collections'].keys():
                name= _known_objects['_collections'][this_name]
            else:
                name= args[0]
        if isinstance(name,(basestring)):
            out= cls(vxvv=_from_name_oneobject(name,obs),radec=True,
                     ro=obs[0],vo=obs[1],zo=obs[2],solarmotion=obs[3])
        else: # assume list
            all_vxvv= []
            for tname in name:
                all_vxvv.append(_from_name_oneobject(tname,obs))
            out= cls(vxvv=all_vxvv,radec=True,
                     ro=obs[0],vo=obs[1],zo=obs[2],solarmotion=obs[3])
        out.name= name
        return out

    @classmethod
    def from_fit(cls,init_vxvv,vxvv,vxvv_err=None,pot=None,
                 radec=False,lb=False,
                 customsky=False,lb_to_customsky=None,
                 pmllpmbb_to_customsky=None,
                 tintJ=10,ntintJ=1000,integrate_method='dopr54_c',
                 ro=None,vo=None,zo=None,solarmotion=None,
                 disp=False):
        """
        NAME:

           from_fit

        PURPOSE:

           Initialize an Orbit using a fit to data

        INPUT:

           init_vxvv - initial guess for the fit (same representation [e.g.,radec=True] as vxvv data, except when customsky, then init_vxvv is assumed to be ra,dec)

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

               lb_to_customsky= function that converts l,b,degree=False to the custom sky coordinates (like lb_to_radec); needs to be given when customsky=True

               pmllpmbb_to_customsky= function that converts pmll,pmbb,l,b,degree=False to proper motions in the custom sky coordinates (like pmllpmbb_to_pmrapmdec); needs to be given when customsky=True

           Keywords related to the orbit integrations:

               tintJ= (default: 10) time to integrate orbits for fitting the orbit (can be Quantity)

               ntintJ= (default: 1000) number of time-integration points

               integrate_method= (default: 'dopr54_c') integration method to use

           Keywords related to the coordinate transformation:

               ro= distance from vantage point to GC (kpc; can be Quantity)

               vo= circular velocity at ro (km/s; can be Quantity)

               zo= offset toward the NGP of the Sun wrt the plane (kpc; can be Quantity; default = 20.8 pc from Bennett & Bovy 2019)

               solarmotion= 'hogg' or 'dehnen', or 'schoenrich', or value in [-U,V,W]; can be Quantity

           disp= (False) display the optimizer's convergence message

        OUTPUT:

           Orbit instance

        HISTORY:

           2014-06-17 - Written - Bovy (IAS)

           2019-05-22 - Incorporated into new Orbit class as from_fit -  Bovy (UofT)

        """
        pot= flatten_potential(pot)
        # Setup Orbit instance for initialization to, among other things, 
        # parse the coordinate-transformation keywords
        init_orbit= cls(init_vxvv,radec=radec or customsky,
                        lb=lb,ro=ro,vo=vo,zo=zo,
                        solarmotion=solarmotion)
        _check_potential_dim(init_orbit,pot)
        _check_consistent_units(init_orbit,pot)
        if radec or lb or customsky:
            obs, ro, vo= _parse_radec_kwargs(init_orbit,
                                             {'ro':init_orbit._ro,
                                              'vo':init_orbit._vo},
                                             vel=True,dontpop=True)
        else:
            obs, ro, vo= None, None, None
        if customsky \
                and (lb_to_customsky is None or pmllpmbb_to_customsky is None):
            raise IOError('if customsky=True, the functions lb_to_customsky and pmllpmbb_to_customsky need to be given')
        new_vxvv, maxLogL= _fit_orbit(init_orbit,vxvv,vxvv_err,pot,
                                      radec=radec,lb=lb,
                                      customsky=customsky,
                                      lb_to_customsky=lb_to_customsky,
                                      pmllpmbb_to_customsky=pmllpmbb_to_customsky,
                                      tintJ=tintJ,ntintJ=ntintJ,
                                      integrate_method=integrate_method,
                                      ro=ro,vo=vo,obs=obs,disp=disp)
        #Setup with these new initial conditions
        return cls(new_vxvv,
                   ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)

    def __len__(self):
        return 1 if self.shape == () else self.shape[0]

    def dim(self):
        """
        NAME:

           dim

        PURPOSE:

           return the dimension of the Orbit

        INPUT:

           (none)

        OUTPUT:

           dimension

        HISTORY:

           2011-02-03 - Written - Bovy (NYU)

        """
        pdim= self.phasedim()
        if pdim == 2:
            return 1
        elif pdim == 3 or pdim == 4:
            return 2
        elif pdim == 5 or pdim == 6:
            return 3

    def phasedim(self):
        """
        NAME:

           phasedim

        PURPOSE:

           return the phase-space dimension of the problem (2 for 1D, 3 for 2D-axi, 4 for 2D, 5 for 3D-axi, 6 for 3D)

        INPUT:

           (none)

        OUTPUT:

           phase-space dimension

        HISTORY:

           2018-12-20 - Written - Bovy (UofT)

        """
        return self.vxvv.shape[-1]

    def __getattr__(self,name):
        """
        NAME:

            __getattr__

        PURPOSE:

            get or evaluate an attribute for this Orbit instance

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
                kwargs['d1']= kwargs.get('d1','t')
                kwargs['d2']= name.split('plot')[1]
                if ('E' in kwargs['d2'] or kwargs['d2'] == 'Jacobi') \
                        and kwargs.pop('normed',False):
                    kwargs['d2']+= 'norm'
                return self.plot(*args,**kwargs)
            # Assign documentation
            if 'E' in name or 'Jacobi' in name:
                Estring= """pot= Potential instance or list of instances in which the orbit was integrated

           normed= if set, plot {quant}(t)/{quant}(0) rather than {quant}(t)

           """.format(quant=name.split('plot')[1])
            else:
                Estring= ''
            _plot.__doc__= """
        NAME:

           plot{quant}

        PURPOSE:

           plot {quant}(t) along the orbit

        INPUT:

           d1= first dimension to plot ('x', 'y', 'R', 'vR', 'vT', 'z', 'vz', ...); can also be an expression, like 'R*vR', or a user-defined function of time (e.g., lambda t: o.R(t) for R)

           {Estring}ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           matplotlib.plot inputs+galpy.util.plot.plot inputs

        OUTPUT:

           sends plot to output device

        HISTORY:

           2019-04-13 - Written - Bovy (UofT)

        """.format(quant=name.split('plot')[1],Estring=Estring)
            return _plot
        else:
            raise AttributeError("'{}' object has no attribute '{}'"\
                                     .format(self.__class__.__name__,
                                             name))

    def __getitem__(self,key):
        """
        NAME:

           __getitem__

        PURPOSE:

           get a subset of this instance's orbits

        INPUT:

           key - slice

        OUTPUT:

           For single item: Orbit instance, for multiple items: another Orbit instance

        HISTORY:

           2018-12-31 - Written - Bovy (UofT)

        """
        indx_array= numpy.arange(self.size).reshape(self.shape)
        indx_array= indx_array[key]
        flat_indx_array= indx_array.flatten()
        orbits_list= self.vxvv[flat_indx_array]
        # Transfer new shape
        shape_kwargs= {}
        shape_kwargs['shape']= indx_array.shape
        # Transfer physical
        physical_kwargs= {}
        physical_kwargs['_roSet']= self._roSet
        physical_kwargs['_voSet']= self._voSet
        physical_kwargs['_ro']= self._ro
        physical_kwargs['_vo']= self._vo
        physical_kwargs['_zo']= self._zo
        physical_kwargs['_solarmotion']= self._solarmotion
        # Also transfer all attributes related to integration
        if hasattr(self,'orbit'):
            integrate_kwargs= {}
            integrate_kwargs['t']= self.t
            integrate_kwargs['_integrate_t_asQuantity']= \
                self._integrate_t_asQuantity
            integrate_kwargs['orbit']= \
                copy.deepcopy(self.orbit[flat_indx_array])
            integrate_kwargs['_pot']= self._pot
        else: integrate_kwargs= None
        return self._from_slice(orbits_list,integrate_kwargs,
                                shape_kwargs,physical_kwargs)

    @classmethod
    def _from_slice(cls,orbits_list,integrate_kwargs,shape_kwargs,
                    physical_kwargs):
        out= cls(vxvv=orbits_list)
        # Set shape
        out.shape= shape_kwargs['shape']
        # Transfer attributes related to physical
        for kw in physical_kwargs:
            out.__dict__[kw]= physical_kwargs[kw]
        # Also transfer all attributes related to integration
        if not integrate_kwargs is None:
            for kw in integrate_kwargs:
                out.__dict__[kw]= integrate_kwargs[kw]
        return out

    def reshape(self,newshape):
        """
        NAME:
        
           reshape

        PURPOSE:

           Change the shape of the Orbit instance

        INPUT:

           newshape - new shape (int or tuple of ints; see numpy.reshape)

        OUTPUT:

           (none; re-shaping is done in-place)

        HISTORY:

           2019-03-20 - Written - Bovy (UofT)

        """
        # We reshape a dummy numpy array to use numpy.reshape's parsing
        dummy= numpy.empty(self.shape)
        try:
            dummy= dummy.reshape(newshape)
        except ValueError:
            # Eventually should just be raise ValueError from None (Python 3)
            raise_from(ValueError('cannot reshape Orbit of shape %s into shape %s'
                                  % (self.shape,newshape)),None)
        self.shape= dummy.shape
        return None

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

           2020-04-22 - Don't turn on a parameter when it is False - Bovy (UofT)

        """
        if not ro is False: self._roSet= True
        if not vo is False: self._voSet= True
        if not ro is None and ro:
            self._ro= conversion.parse_length_kpc(ro)
        if not vo is None and vo:
            self._vo= conversion.parse_velocity_kms(vo)
        return None

    def integrate(self,t,pot,method='symplec4_c',dt=None,numcores=_NUMCORES,
                  force_map=False):
        """
        NAME:

            integrate

        PURPOSE:

            integrate this Orbit instance with multiprocessing

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
                     'dop853' for a 8-5-3 Dormand-Prince integrator in Python
                     'dop853_c' for a 8-5-3 Dormand-Prince integrator in C

            dt - if set, force the integrator to use this basic stepsize; must be an integer divisor of output stepsize (only works for the C integrators that use a fixed stepsize) (can be Quantity)

            numcores - number of cores to use for Python-based multiprocessing (pure Python or using force_map=True); default = OMP_NUM_THREADS

            force_map= (False) if True, force use of Python-based multiprocessing (not recommended)

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
            t= conversion.parse_time(t,ro=self._ro,vo=self._vo)
        else: self._integrate_t_asQuantity= False
        if _APY_LOADED and not dt is None and isinstance(dt,units.Quantity):
            dt= conversion.parse_time(dt,ro=self._ro,vo=self._vo)
        from ..potential import MWPotential
        if pot == MWPotential:
            warnings.warn("Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                          galpyWarning)
        if not _check_integrate_dt(t,dt):
            raise ValueError('dt input (integrator stepsize) for Orbit.integrate must be an integer divisor of the output stepsize')
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
            if self.dim() == 1:
                out, msg= integrateLinearOrbit(self._pot,self.vxvv,t,
                                               method,numcores=numcores,dt=dt)
            elif self.dim() == 2:
                out, msg= integratePlanarOrbit(self._pot,self.vxvv,t,
                                               method,numcores=numcores,dt=dt)
            else:
                out, msg= integrateFullOrbit(self._pot,self.vxvv,t,
                                             method,numcores=numcores,dt=dt)
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
        # Check whether r ever < minr if dynamical friction is included
        # and warn if so
        # or if using interpSphericalPotential and r < rmin or r > rmax
        from ..potential import ChandrasekharDynamicalFrictionForce, \
            interpSphericalPotential
        if numpy.any([isinstance(p,ChandrasekharDynamicalFrictionForce)
                      for p in flatten_potential([pot])]): # make sure pot=list
            lpot= flatten_potential([pot])
            cdf_indx= numpy.arange(len(lpot))[\
                numpy.array([isinstance(p,ChandrasekharDynamicalFrictionForce)
                             for p in lpot],dtype='bool')][0]
            if numpy.any(self.r(self.t,use_physical=False) \
                             < lpot[cdf_indx]._minr):
                warnings.warn("""Orbit integration with """
                              """ChandrasekharDynamicalFrictionForce """
                              """entered domain where r < minr and """
                              """ChandrasekharDynamicalFrictionForce is """
                              """turned off; initialize """
                              """ChandrasekharDynamicalFrictionForce with a """
                              """smaller minr to avoid this if you wish """
                              """(but note that you want to turn it off """
                              """close to the center for an object that """
                              """sinks all the way to r=0, to avoid """
                              """numerical instabilities)""",
                          galpyWarning)
        elif numpy.any([isinstance(p,interpSphericalPotential)
                      for p in flatten_potential([pot])]): # make sure pot=list
            lpot= flatten_potential([pot])
            isp_indx= numpy.arange(len(lpot))[\
                numpy.array([isinstance(p,interpSphericalPotential)
                             for p in lpot],dtype='bool')][0]
            if numpy.any(self.r(self.t,use_physical=False) \
                             < lpot[isp_indx]._rmin) \
                             or numpy.any(self.r(self.t,use_physical=False) \
                                          > lpot[isp_indx]._rmax):
                warnings.warn("""Orbit integration with """
                              """interpSphericalPotential visited radii """
                              """outside of the interpolation range; """
                              """initialize interpSphericalPotential """
                              """with a wider radial range to avoid this """
                              """if you wish (min/max r = {:.3f},{:.3f}"""\
                              .format(self.rperi(),self.rap()),
                          galpyWarning)
        return None

    def integrate_dxdv(self,dxdv,t,pot,method='dopr54_c',dt=None,
                       numcores=_NUMCORES,force_map=False,
                       rectIn=False,rectOut=False):
        """
        NAME:

           integrate_dxdv

        PURPOSE:

           integrate the orbit and a small area of phase space

        INPUT:

           dxdv - [dR,dvR,dvT,dphi], shape=(*input_shape,4)

           t - list of times at which to output (0 has to be in this!) (can be Quantity)

           pot - potential instance or list of instances

           dt - if set, force the integrator to use this basic stepsize; must be an integer divisor of output stepsize (only works for the C integrators that use a fixed stepsize) (can be Quantity)

           method = 'odeint' for scipy's odeint
                    'rk4_c' for a 4th-order Runge-Kutta integrator in C
                    'rk6_c' for a 6-th order Runge-Kutta integrator in C
                    'dopr54_c' for a 5-4 Dormand-Prince integrator in C
                    'dopr853_c' for a 8-5-3 Dormand-Prince integrator in C

           rectIn= (False) if True, input dxdv is in rectangular coordinates

           rectOut= (False) if True, output dxdv (that in orbit_dxdv) is in rectangular coordinates

           numcores - number of cores to use for Python-based multiprocessing (pure Python or using force_map=True); default = OMP_NUM_THREADS

           force_map= (False) if True, force use of Python-based multiprocessing (not recommended)

        OUTPUT:

           (none) (get the actual orbit using getOrbit_dxdv(), the orbit that is integrated alongside with dxdv is stored as usual, any previous regular orbit integration will be erased!)

        HISTORY:

           2011-10-17 - Written - Bovy (IAS)

           2014-06-29 - Added rectIn and rectOut - Bovy (IAS)

           2019-05-21 - Parallelized and incorporated into new Orbits class - Bovy (UofT)

        """
        if not self.phasedim() == 4:
            raise AttributeError('integrate_dxdv is only implemented for 4D (planar) orbits')
        if method.lower() not in ['odeint', 'dop853', 'rk4_c', 'rk6_c',
                                  'dopr54_c', 'dop853_c']:
            if 'leapfrog' in method.lower() or 'symplec' in method.lower():
                raise ValueError('{:s} is not a valid `method for integrate_dxdv, because symplectic integrators cannot be used`'.format(method))
            else:
                raise ValueError('{:s} is not a valid `method for integrate_dxdv`'.format(method))
        pot= flatten_potential(pot)
        _check_potential_dim(self,pot)
        _check_consistent_units(self,pot)
        # Parse t
        if _APY_LOADED and isinstance(t,units.Quantity):
            self._integrate_t_asQuantity= True
            t= conversion.parse_time(t,ro=self._ro,vo=self._vo)
        else: self._integrate_t_asQuantity= False
        if not dt is None:
            dt= conversion.parse_time(dt,ro=self._ro,vo=self._vo)
        # Parse dxdv
        dxdv= numpy.array(dxdv)
        if dxdv.ndim > 1:
            dxdv= dxdv.reshape((numpy.prod(dxdv.shape[:-1]),dxdv.shape[-1])) 
        else:
            dxdv= numpy.atleast_2d(dxdv)
        # Delete attributes for interpolation and rperi etc. determination
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        if self.dim() == 2:
            thispot= toPlanarPotential(pot)
        self.t= numpy.array(t)
        self._pot_dxdv= thispot
        self._pot= thispot
        #First check that the potential has C
        if '_c' in method:
            allHasC= _check_c(pot) and _check_c(pot,dxdv=True)
            if not ext_loaded or \
                    (not allHasC and not 'leapfrog' in method and not 'symplec' in method):
                method= 'odeint'
                if not ext_loaded: # pragma: no cover
                    warnings.warn("Cannot use C integration because C extension not loaded (using %s instead)" % (method), galpyWarning)
                else:
                    warnings.warn("Using odeint because not all used potential have adequate C implementations to integrate phase-space volumes",galpyWarning)
        # Implementation with parallel_map in Python
        if True or not '_c' in method or not ext_loaded or force_map:
            if self.dim() == 2:
                out, msg= integratePlanarOrbit_dxdv(self._pot,self.vxvv,dxdv,
                                                    t,method,rectIn,rectOut,
                                                    numcores=numcores,dt=dt)
        # Store orbit internally
        self.orbit_dxdv= out
        self.orbit= self.orbit_dxdv[...,:4]
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

           2019-03-02 - Written - Bovy (UofT)

        """
        if inplace:
            self.vxvv[...,1]= -self.vxvv[...,1]
            if self.phasedim() > 2:
                self.vxvv[...,2]= -self.vxvv[...,2]
            if self.phasedim() > 4:
                self.vxvv[...,4]= -self.vxvv[...,4]
            if hasattr(self,'orbit'):
                self.orbit[...,1]= -self.orbit[...,1]
                if self.phasedim() > 2:
                    self.orbit[...,2]= -self.orbit[...,2]
                if self.phasedim() > 4:
                    self.orbit[...,4]= -self.orbit[...,4]
                if hasattr(self,"_orbInterp"):
                    delattr(self,"_orbInterp")
            return None
        orbSetupKwargs= {'ro':self._ro,
                         'vo':self._vo,
                         'zo':self._zo,
                         'solarmotion':self._solarmotion}
        if self.phasedim() == 2:
            orbSetupKwargs.pop('zo',None)
            orbSetupKwargs.pop('solarmotion',None)
            out= Orbit(numpy.array([self.vxvv[...,0],-self.vxvv[...,1]]).T,
                       **orbSetupKwargs)
        elif self.phasedim() == 3:
            out= Orbit(numpy.array([self.vxvv[...,0],-self.vxvv[...,1],
                                    -self.vxvv[...,2]]).T,**orbSetupKwargs)
        elif self.phasedim() == 4:
            out= Orbit(numpy.array([self.vxvv[...,0],-self.vxvv[...,1],
                                    -self.vxvv[...,2],self.vxvv[...,3]]).T,
                       **orbSetupKwargs)
        elif self.phasedim() == 5:
            out= Orbit(numpy.array([self.vxvv[...,0],-self.vxvv[...,1],
                                    -self.vxvv[...,2],self.vxvv[...,3],
                                    -self.vxvv[...,4]]).T,**orbSetupKwargs)
        elif self.phasedim() == 6:
            out= Orbit(numpy.array([self.vxvv[...,0],-self.vxvv[...,1],
                                    -self.vxvv[...,2],self.vxvv[...,3],
                                    -self.vxvv[...,4],self.vxvv[...,5]]).T,
                       **orbSetupKwargs)
        out._roSet= self._roSet
        out._voSet= self._voSet
        # Make sure the output has the same shape as the original Orbit
        out.reshape(self.shape)
        return out
        
    @shapeDecorator
    def getOrbit(self):
        """

        NAME:

           getOrbit

        PURPOSE:

           return previously calculated orbits

        INPUT:

           (none)

        OUTPUT:

           array orbit[*input_shape,nt,nphasedim]

        HISTORY:

           2019-03-02 - Written - Bovy (UofT)

        """
        return self.orbit.copy()

    @shapeDecorator
    def getOrbit_dxdv(self):
        """

        NAME:

           getOrbit_dxdv

        PURPOSE:

           return a previously calculated integration of a small phase-space volume (with integrate_dxdv)

        INPUT:

           (none)

        OUTPUT:

           array orbit[*input_shape,nt,nphasedim]

        HISTORY:

           2019-05-21 - Written - Bovy (UofT)

        """
        return self.orbit_dxdv[...,4:].copy()

    @physical_conversion('energy')
    @shapeDecorator
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

           energy [*input_shape,nt]

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
                out= (evaluatelinearPotentials(\
                        pot,thiso[0],
                        t=numpy.tile(t,thiso[0].T.shape[:-1]+(1,)).T,
                        use_physical=False)\
                          +thiso[1]**2./2.).T
            except (ValueError,TypeError,IndexError):
                out= (numpy.array([[evaluatelinearPotentials(\
                                    pot,thiso[0][ii][jj],t=t[ii],
                                    use_physical=False)
                                    for ii in range(len(thiso[0]))]
                                   for jj in range(self.size)])\
                          +(thiso[1]**2./2.).T)
        elif self.phasedim() == 3:
            try:
                out= (evaluateplanarPotentials(\
                        pot,thiso[0],
                        t=numpy.tile(t,thiso[0].T.shape[:-1]+(1,)).T,
                        use_physical=False)\
                          +thiso[1]**2./2.+thiso[2]**2./2.).T
            except (ValueError,TypeError,IndexError):
                out= (numpy.array([[evaluateplanarPotentials(\
                                    pot,thiso[0][ii][jj],t=t[ii],
                                    use_physical=False)
                                    for ii in range(len(thiso[0]))]
                                   for jj in range(self.size)])
                      +(thiso[1]**2./2.+thiso[2]**2./2.).T)
        elif self.phasedim() == 4:
            try:
                out= (evaluateplanarPotentials(\
                        pot,thiso[0],phi=thiso[-1],
                        t=numpy.tile(t,thiso[0].T.shape[:-1]+(1,)).T,
                        use_physical=False)\
                          +thiso[1]**2./2.+thiso[2]**2./2.).T
            except (ValueError,TypeError,IndexError):
                out= (numpy.array([[evaluateplanarPotentials(\
                                    pot,thiso[0][ii][jj],t=t[ii],
                                    phi=thiso[-1][ii][jj],
                                    use_physical=False)
                                    for ii in range(len(thiso[0]))]
                                   for jj in range(self.size)])
                                  +(thiso[1]**2./2.+thiso[2]**2./2.).T)
        elif self.phasedim() == 5:
            z= kwargs.get('_z',1.)*thiso[3] # For ER and Ez
            vz= kwargs.get('_vz',1.)*thiso[4] # For ER and Ez
            try:
                out= (evaluatePotentials(\
                        pot,thiso[0],z,
                        t=numpy.tile(t,thiso[0].T.shape[:-1]+(1,)).T,
                        use_physical=False)\
                          +thiso[1]**2./2.+thiso[2]**2./2.+vz**2./2.).T
            except (ValueError,TypeError,IndexError):
                out= (numpy.array([[evaluatePotentials(\
                                    pot,thiso[0][ii][jj],
                                    z[ii][jj],
                                    t=t[ii],
                                    use_physical=False)
                                    for ii in range(len(thiso[0]))]
                                   for jj in range(self.size)])
                      +(thiso[1]**2./2.+thiso[2]**2./2.+vz**2./2.).T)
        elif self.phasedim() == 6:
            z= kwargs.get('_z',1.)*thiso[3] # For ER and Ez
            vz= kwargs.get('_vz',1.)*thiso[4] # For ER and Ez
            try:
                out= (evaluatePotentials(\
                        pot,thiso[0],z,phi=thiso[-1],
                        t=numpy.tile(t,thiso[0].T.shape[:-1]+(1,)).T,
                        use_physical=False)\
                          +thiso[1]**2./2.+thiso[2]**2./2.+vz**2./2.).T
            except (ValueError,TypeError,IndexError):
                out= (numpy.array([[evaluatePotentials(\
                                    pot,thiso[0][ii][jj],
                                    z[ii][jj],
                                    t=t[ii],
                                    phi=thiso[-1][ii][jj],
                                    use_physical=False)
                                    for ii in range(len(thiso[0]))]
                                   for jj in range(self.size)])
                      +(thiso[1]**2./2.+thiso[2]**2./2.+vz**2./2.).T)
        if onet:
            return out[:,0]
        else:
            return out

    @physical_conversion('action')
    @shapeDecorator
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

           angular momentum [*input_shape,nt,3]

        HISTORY:

           2019-03-01 - Written - Bovy (UofT)

        """
        if self.dim() == 1:
            raise AttributeError("'linear Orbit has no angular momentum")
        #Get orbit
        if self.dim() == 2:
            thiso= self._call_internal(*args,**kwargs)
            return (thiso[0]*thiso[2]).T
        elif self.phasedim() == 5:
            raise AttributeError("You must track the azimuth to get the angular momentum of a 3D Orbit")
        else: # phasedim == 6
            old_physical= kwargs.get('use_physical',None)
            kwargs['use_physical']= False
            kwargs['dontreshape']= True
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
            if not old_physical is None:
                kwargs['use_physical']= old_physical
            else:
                kwargs.pop('use_physical')
            kwargs.pop('dontreshape')
            return out
        
    @physical_conversion('action')
    @shapeDecorator
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

           z-component of the angular momentum [*input_shape,nt]

        HISTORY:

           2019-03-01 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        return (thiso[0]*thiso[2]).T

    @physical_conversion('energy')
    @shapeDecorator
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

           radial energy [*input_shape,nt]

        HISTORY:

           2019-03-01 - Written - Bovy (UofT)

        """
        old_physical= kwargs.get('use_physical',None)
        kwargs['use_physical']= False
        kwargs['_z']= 0.
        kwargs['_vz']= 0.
        kwargs['dontreshape']= True
        out= self.E(*args,**kwargs)
        if not old_physical is None:
            kwargs['use_physical']= old_physical
        else:
            kwargs.pop('use_physical')
        kwargs.pop('_z')
        kwargs.pop('_vz')
        kwargs.pop('dontreshape')
        return out

    @physical_conversion('energy')
    @shapeDecorator
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

           vertical energy [*input_shape,nt]

        HISTORY:

           2019-03-01 - Written - Bovy (UofT)

        """
        old_physical= kwargs.get('use_physical',None)
        kwargs['use_physical']= False
        kwargs['dontreshape']= True
        tE= self.E(*args,**kwargs)
        kwargs['_z']= 0.
        kwargs['_vz']= 0.
        out= tE-self.E(*args,**kwargs)
        if not old_physical is None:
            kwargs['use_physical']= old_physical
        else:
            kwargs.pop('use_physical')
        kwargs.pop('_z')
        kwargs.pop('_vz')
        kwargs.pop('dontreshape')
        return out

    @physical_conversion('energy')
    @shapeDecorator
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

           Jacobi integral [*input_shape,nt]

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
        OmegaP= conversion.parse_frequency(OmegaP,ro=self._ro,vo=self._vo)
        #Make sure you are not using physical coordinates
        old_physical= kwargs.get('use_physical',None)
        kwargs['use_physical']= False
        kwargs['dontreshape']= True
        if not isinstance(OmegaP,(int,float)) and len(OmegaP) == 3:
            if isinstance(OmegaP,list): thisOmegaP= numpy.array(OmegaP)
            else: thisOmegaP= OmegaP
            tL= self.L(*args,**kwargs)
            if len(tL.shape) == 2: # 1 time
                out= self.E(*args,**kwargs)-numpy.einsum('i,ji->j',
                                                         thisOmegaP,tL)
            else:
                out= self.E(*args,**kwargs)-numpy.einsum('i,jki->jk',
                                                         thisOmegaP,tL)
        else:
            out= self.E(*args,**kwargs)-OmegaP*self.Lz(*args,**kwargs)
        if not old_physical is None:
            kwargs['use_physical']= old_physical
        else:
            kwargs.pop('use_physical')
        kwargs.pop('dontreshape')
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
        from .. import actionAngle
        if not pot is None: pot= flatten_potential(pot)
        if self.dim() == 2 and (type == 'staeckel' or type == 'adiabatic'):
            # No reason to do Staeckel or adiabatic...
            type= 'spherical'
        elif self.dim() == 1:
            raise RuntimeError("Orbit action-angle methods are not supported for 1D orbits")
        delta= kwargs.pop('delta',None)
        if not delta is None:
            delta= conversion.parse_length(delta,ro=self._ro)
        b= kwargs.pop('b',None)
        if not b is None:
            b= conversion.parse_length(b,ro=self._ro)
        if pot is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
        if hasattr(self,'_aA'):
            if (not pot is None and pot != self._aAPot) \
                    or (not type is None and type != self._aAType) \
                    or (not delta is None and hasattr(self._aA,'_delta') 
                        and numpy.any(delta != self._aA._delta)) \
                    or (delta is None
                        and hasattr(self,'_aA_delta_automagic')
                        and not self._aA_delta_automagic) \
                    or (not b is None and hasattr(self._aA,'_aAI') 
                        and numpy.any(b != self._aA._aAI.b)) \
                    or ('ip' in kwargs and hasattr(self._aA,'_aAI') 
                        and (numpy.any(kwargs['ip'].b != self._aA._aAI.b) \
                        or numpy.any(kwargs['ip']._amp != self._aA._aAI.amp))):
                for attr in list(self.__dict__):
                    if '_aA' in attr: delattr(self,attr)
            else:
                return None
        _check_consistent_units(self,pot)
        self._aAPot= pot
        self._aAType= type
        #Setup
        if self._aAType.lower() == 'adiabatic':
            self._aA= actionAngle.actionAngleAdiabatic(pot=self._aAPot,
                                                       **kwargs)
        elif self._aAType.lower() == 'staeckel':
            # try to make sure this is not 0
            tz= self.z(use_physical=False,dontreshape=True)\
                +(numpy.fabs(self.z(use_physical=False,
                                    dontreshape=True)) < 1e-8) \
                * (2.*(self.z(use_physical=False,
                              dontreshape=True) >= 0)-1.)*1e-10
            self._aA_delta_automagic= False
            if delta is None:
                self._aA_delta_automagic= True
                try:
                    delta= actionAngle.estimateDeltaStaeckel(\
                        self._aAPot,
                        self.R(use_physical=False,dontreshape=True),
                        tz,no_median=True,use_physical=False)
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
            from ..actionAngle import actionAngleIsochroneApprox
            self._aA= actionAngleIsochroneApprox(pot=self._aAPot,b=b,
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
            tz= self.z(use_physical=False,dontreshape=True)\
                +(numpy.fabs(self.z(use_physical=False,
                                    dontreshape=True)) < 1e-8) \
                * (2.*(self.z(use_physical=False,
                              dontreshape=True) >= 0)-1.)*1e-10
            tvz= self.vz(use_physical=False,dontreshape=True)
        elif self.dim() == 2:
            tz= numpy.zeros(self.size)
            tvz= numpy.zeros(self.size)
        # self.dim() == 1 error caught by _setupaA
        self._aA_ecc, self._aA_zmax, self._aA_rperi, self._aA_rap=\
            self._aA.EccZmaxRperiRap(self.R(use_physical=False,
                                            dontreshape=True),
                                     self.vR(use_physical=False,
                                             dontreshape=True),
                                     self.vT(use_physical=False,
                                             dontreshape=True),
                                     tz,tvz,
                                     use_physical=False)
        return None        

    def _setup_actionsFreqsAngles(self,pot=None,**kwargs):
        """Internal function to compute the actions, frequencies, and angles and cache them for re-use"""
        self._setupaA(pot=pot,**kwargs)
        if hasattr(self,'_aA_jr'): return None
        if self.dim() == 3:
            # try to make sure this is not 0
            tz= self.z(use_physical=False,dontreshape=True)\
                +(numpy.fabs(self.z(use_physical=False,
                                    dontreshape=True)) < 1e-8) \
                * (2.*(self.z(use_physical=False,
                              dontreshape=True) >= 0)-1.)*1e-10
            tvz= self.vz(use_physical=False,dontreshape=True)
        elif self.dim() == 2:
            tz= numpy.zeros(self.size)
            tvz= numpy.zeros(self.size)
        # self.dim() == 1 error caught by _setupaA
        self._aA_jr, self._aA_jp, self._aA_jz, \
            self._aA_Or, self._aA_Op, self._aA_Oz, \
            self._aA_wr, self._aA_wp, self._aA_wz= \
               self._aA.actionsFreqsAngles(\
                  self.R(use_physical=False,dontreshape=True),
                  self.vR(use_physical=False,dontreshape=True),
                  self.vT(use_physical=False,dontreshape=True),
                  tz,tvz,
                  self.phi(use_physical=False,dontreshape=True),
                  use_physical=False)
        return None

    @shapeDecorator
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

           eccentricity [*input_shape]

        HISTORY:

           2019-02-25 - Written - Bovy (UofT)

        """
        if analytic:
            self._setup_EccZmaxRperiRap(pot=pot,**kwargs)
            return self._aA_ecc
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        rs= self.r(self.t,use_physical=False,dontreshape=True)
        return (numpy.amax(rs,axis=-1)-numpy.amin(rs,axis=-1))\
            /(numpy.amax(rs,axis=-1)+numpy.amin(rs,axis=-1))

    @physical_conversion('position')
    @shapeDecorator
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

           R_ap [*input_shape]

        HISTORY:

           2019-02-25 - Written - Bovy (UofT)

        """
        if analytic:
            self._setup_EccZmaxRperiRap(pot=pot,**kwargs)
            return self._aA_rap
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        rs= self.r(self.t,use_physical=False,dontreshape=True)
        return numpy.amax(rs,axis=-1)

    @physical_conversion('position')
    @shapeDecorator
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

           R_peri [*input_shape]

        HISTORY:

           2019-02-25 - Written - Bovy (UofT)

        """
        if analytic:
            self._setup_EccZmaxRperiRap(pot=pot,**kwargs)
            return self._aA_rperi
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        rs= self.r(self.t,use_physical=False,dontreshape=True)
        return numpy.amin(rs,axis=-1)

    @physical_conversion('position')
    @shapeDecorator
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

           R_guiding [*input_shape,nt]

        HISTORY:

           2019-03-02 - Written as thin wrapper around Potential.rl - Bovy (UofT)

        """
        pot= kwargs.get('pot',self.__dict__.get('_pot',None))
        if pot is None:
            raise RuntimeError("You need to specify the potential as pot= to compute the guiding-center radius")
        flatten_potential(pot)
        if _isNonAxi(pot):
            raise RuntimeError('Potential given to rguiding is non-axisymmetric, but rguiding requires an axisymmetric potential')
        _check_consistent_units(self,pot)
        Lz= numpy.atleast_1d(self.Lz(*args,use_physical=False,
                                      dontreshape=True))
        Lz_shape= Lz.shape
        Lz= Lz.flatten()
        if len(Lz) > 500:
            # Build interpolation grid, 500 ~ 1s
            precomputergLzgrid= numpy.linspace(numpy.nanmin(Lz),
                                               numpy.nanmax(Lz),
                                               500)
            rls= numpy.array([rl(pot,lz,use_physical=False)
                              for lz in precomputergLzgrid])
            #Spline interpolate
            return interpolate.InterpolatedUnivariateSpline(\
                precomputergLzgrid,rls,k=3)(Lz).reshape(Lz_shape)
        else:
            return numpy.array([rl(pot,lz,use_physical=False)
                                for lz in Lz]).reshape(Lz_shape)

    @physical_conversion('position')
    @shapeDecorator
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

           Z_max [*input_shape]

        HISTORY:

           2019-02-25 - Written - Bovy (UofT)

        """
        if analytic:
            self._setup_EccZmaxRperiRap(pot=pot,**kwargs)
            return self._aA_zmax
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        return numpy.amax(numpy.fabs(self.z(self.t,use_physical=False,
                                            dontreshape=True)),
                          axis=-1)

    @physical_conversion('action')
    @shapeDecorator
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

           jr [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_jr

    @physical_conversion('action')
    @shapeDecorator
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

           jp [*input_shape]

        HISTORY:

           2019-02-26 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_jp

    @physical_conversion('action')
    @shapeDecorator
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

           jz [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_jz

    @physical_conversion('angle')
    @shapeDecorator
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

           wr [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_wr

    @physical_conversion('angle')
    @shapeDecorator
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

           wp [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_wp

    @physical_conversion('angle')
    @shapeDecorator
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

           wz [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_wz

    @physical_conversion('time')
    @shapeDecorator
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

           Tr [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return 2.*numpy.pi/self._aA_Or

    @physical_conversion('time')
    @shapeDecorator
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

           Tp [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return 2.*numpy.pi/self._aA_Op

    @shapeDecorator
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

           Tr/Tp*pi [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_Op/self._aA_Or*numpy.pi
 
    @physical_conversion('time')
    @shapeDecorator
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

           Tz [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return 2.*numpy.pi/self._aA_Oz

    @physical_conversion('frequency')
    @shapeDecorator
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

           Or [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_Or

    @physical_conversion('frequency')
    @shapeDecorator
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

           Op [*input_shape]

        HISTORY:

           2019-02-27 - Written - Bovy (UofT)

        """
        self._setup_actionsFreqsAngles(pot=pot,**kwargs)
        return self._aA_Op

    @physical_conversion('frequency')
    @shapeDecorator
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

           Oz [*input_shape]

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
                return self.t.copy()
            except AttributeError:
                return 0.
        else:
            out= args[0]
            return conversion.parse_time(out,ro=self._ro,vo=self._vo)

    @physical_conversion('position')
    @shapeDecorator
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

           R(t) [*input_shape,nt]

        HISTORY:

           2019-02-01 - Written - Bovy (UofT)

        """
        return self._call_internal(*args,**kwargs)[0].T

    @physical_conversion('position')
    @shapeDecorator
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

           r(t) [*input_shape,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        if self.dim() == 3:
            return numpy.sqrt(thiso[0]**2.+thiso[3]**2.).T
        else:
            return numpy.fabs(thiso[0]).T

    @physical_conversion('velocity')
    @shapeDecorator
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

           vR(t) [*input_shape,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        return self._call_internal(*args,**kwargs)[1].T

    @physical_conversion('velocity')
    @shapeDecorator
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

           vT(t) [*input_shape,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        return self._call_internal(*args,**kwargs)[2].T

    @physical_conversion('position')
    @shapeDecorator
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

           z(t) [*input_shape,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        if self.dim() < 3:
            raise AttributeError("linear and planar orbits do not have z()")
        return self._call_internal(*args,**kwargs)[3].T

    @physical_conversion('velocity')
    @shapeDecorator
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

           vz(t) [*input_shape,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        if self.dim() < 3:
            raise AttributeError("linear and planar orbits do not have vz()")
        return self._call_internal(*args,**kwargs)[4].T
        
    @physical_conversion('angle')
    @shapeDecorator
    def phi(self,*args,**kwargs):
        """
        NAME:

           phi

        PURPOSE:

           return azimuth

        INPUT:

           t - (optional) time at which to get the azimuth

        OUTPUT:

           phi(t) [*input_shape,nt] in [-pi,pi]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        if self.phasedim() != 4 and self.phasedim() != 6:
            raise AttributeError("Orbit must track azimuth to use phi()")
        return self._call_internal(*args,**kwargs)[-1].T

    @physical_conversion('position')
    @shapeDecorator
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

           x(t) [*input_shape,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        if self.dim() == 1:
            return thiso[0].T
        elif self.phasedim()  != 4 and self.phasedim() != 6:
            raise AttributeError("Orbit must track azimuth to use x()")
        else:
            return (thiso[0]*numpy.cos(thiso[-1,:])).T

    @physical_conversion('position')
    @shapeDecorator
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

           y(t) [*input_shape,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        if self.phasedim()  != 4 and self.phasedim() != 6:
            raise AttributeError("Orbit must track azimuth to use y()")
        else:
            return (thiso[0]*numpy.sin(thiso[-1,:])).T

    @physical_conversion('velocity')
    @shapeDecorator
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

           vx(t) [*input_shape,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        if self.dim() == 1:
            return thiso[1].T
        elif self.phasedim()  != 4 and self.phasedim() != 6:
            raise AttributeError("Orbit must track azimuth to use vx()")
        else:
            return (thiso[1]*numpy.cos(thiso[-1])
                    -thiso[2]*numpy.sin(thiso[-1])).T

    @physical_conversion('velocity')
    @shapeDecorator
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

           vy(t) [*input_shape,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        if self.phasedim()  != 4 and self.phasedim() != 6:
            raise AttributeError("Orbit must track azimuth to use vy()")
        else:
            return (thiso[2]*numpy.cos(thiso[-1])
                    +thiso[1]*numpy.sin(thiso[-1])).T

    @physical_conversion('frequency-kmskpc')
    @shapeDecorator
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

           vphi(t) [*input_shape,nt]

        HISTORY:

           2019-02-20 - Written - Bovy (UofT)

        """
        thiso= self._call_internal(*args,**kwargs)
        return (thiso[2]/thiso[0]).T
    
    @physical_conversion('velocity')
    @shapeDecorator
    def vr(self,*args,**kwargs):
        """
        NAME:

           vr

        PURPOSE:

           return spherical radial velocity. For < 3 dimensions returns vR

        INPUT:

           t - (optional) time at which to get the radial velocity

           vo= (Object-wide default) physical scale for velocities to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vr(t) [*input_shape,nt]

        HISTORY:

           2020-07-01 - Written - James Lane (UofT)

        """
        thiso = self._call_internal(*args,**kwargs)
        if self.dim() == 3:
            r = numpy.sqrt(thiso[0]**2.+thiso[3]**2.)
            return ((thiso[0]*thiso[1]+thiso[3]*thiso[4])/r).T
        else:
            return thiso[1].T
    
    @physical_conversion('velocity')
    @shapeDecorator
    def vtheta(self,*args,**kwargs):
        """
        NAME:

           vtheta
           
        PURPOSE:

           return spherical polar velocity

        INPUT:

           t - (optional) time at which to get the theta velocity

           vo= (Object-wide default) physical scale for velocities to use to convert

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           vtheta(t) [*input_shape,nt]

        HISTORY:

           2020-07-01 - Written - James Lane (UofT)

        """
        thiso = self._call_internal(*args,**kwargs)
        if not self.dim() == 3:
            raise AttributeError("Orbit must be 3D to use vtheta()")
        else:
            r = numpy.sqrt( thiso[0]**2.+thiso[3]**2.)
            return ((thiso[1]*thiso[3]-thiso[0]*thiso[4])/r).T
            
    @physical_conversion('angle')
    @shapeDecorator
    def theta(self,*args,**kwargs):
        """
        NAME:

           theta
           
        PURPOSE:

           return spherical polar angle

        INPUT:

           t - (optional) time at which to get the angle

        OUTPUT:

           theta(t) [*input_shape,nt]

        HISTORY:

           2020-07-01 - Written - James Lane (UofT)

        """
        thiso = self._call_internal(*args,**kwargs)
        if self.dim() != 3:
            raise AttributeError("Orbit must be 3D to use theta()")
        else:
            return numpy.arctan2(thiso[0],thiso[3])
    
    
    @physical_conversion('angle_deg')
    @shapeDecorator
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

           ra(t) [*input_shape,nt] 

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
    @shapeDecorator
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

           dec(t) [*input_shape,nt] 

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
    @shapeDecorator
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

           l(t) [*input_shape,nt] 

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'ll')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _lbd(self,thiso,*args,**kwargs).T[0].reshape(thiso_shape[1:]).T

    @physical_conversion('angle_deg')
    @shapeDecorator
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

           b(t) [*input_shape,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT

        """
        _check_roSet(self,kwargs,'bb')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _lbd(self,thiso,*args,**kwargs).T[1].reshape(thiso_shape[1:]).T

    @physical_conversion('position_kpc')
    @shapeDecorator
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

           dist(t) in kpc [*input_shape,nt]

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'dist')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        return _lbd(self,thiso,*args,**kwargs).T[2].reshape(thiso_shape[1:]).T

    @physical_conversion('proper-motion_masyr')
    @shapeDecorator
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

           pm_ra(t) in mas / yr [*input_shape,nt]

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
    @shapeDecorator
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

           pm_dec(t) in mas/yr [*input_shape,nt]

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
    @shapeDecorator
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

           pm_l(t) in mas/yr [*input_shape,nt]

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
    @shapeDecorator
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

           pm_b(t) in mas/yr [*input_shape,nt]

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
    @shapeDecorator
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

           vlos(t) in km/s [*input_shape,nt]

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

    @shapeDecorator
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

           v_ra(t) in km/s [*input_shape]

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'vra')
        _check_voSet(self,kwargs,'vra')
        kwargs['dontreshape']= True
        dist= self.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            result= units.Quantity(dist.to(units.kpc).value*_K*
                                   self.pmra(*args,**kwargs)\
                                       .to(units.mas/units.yr).value,
                                   unit=units.km/units.s)
        else:
            result= dist*_K*self.pmra(*args,**kwargs)
        kwargs.pop('dontreshape')
        return result

    @shapeDecorator
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

           v_dec(t) in km/s [*input_shape]

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'vdec')
        _check_voSet(self,kwargs,'vdec')
        kwargs['dontreshape']= True
        dist= self.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            result= units.Quantity(dist.to(units.kpc).value*_K*
                                   self.pmdec(*args,**kwargs)\
                                       .to(units.mas/units.yr).value,
                                   unit=units.km/units.s)
        else:
            result= dist*_K*self.pmdec(*args,**kwargs)
        kwargs.pop('dontreshape')
        return result

    @shapeDecorator
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

           v_l(t) in km/s [*input_shape]

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'vll')
        _check_voSet(self,kwargs,'vll')
        kwargs['dontreshape']= True
        dist= self.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            result= units.Quantity(dist.to(units.kpc).value*_K*
                                   self.pmll(*args,**kwargs)\
                                       .to(units.mas/units.yr).value,
                                   unit=units.km/units.s)
        else:
            result= dist*_K*self.pmll(*args,**kwargs)
        kwargs.pop('dontreshape')
        return result
        
    @shapeDecorator
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

           v_b(t) in km/s [*input_shape]

        HISTORY:

           2019-02-28 - Written - Bovy (UofT)

        """
        _check_roSet(self,kwargs,'vbb')
        _check_voSet(self,kwargs,'vbb')
        kwargs['dontreshape']= True
        dist= self.dist(*args,**kwargs)
        if _APY_UNITS and isinstance(dist,units.Quantity):
            result= units.Quantity(dist.to(units.kpc).value*_K*
                                   self.pmbb(*args,**kwargs)\
                                       .to(units.mas/units.yr).value,
                                   unit=units.km/units.s)
        else:
            result= dist*_K*self.pmbb(*args,**kwargs)
        kwargs.pop('dontreshape')
        return result

    @physical_conversion('position_kpc')
    @shapeDecorator
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

           helioX(t) in kpc [*input_shape,nt]

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
    @shapeDecorator
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

           helioY(t) in kpc [*input_shape,nt]

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
    @shapeDecorator
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

           helioZ(t) in kpc [*input_shape,nt]

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
    @shapeDecorator
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

           U(t) in km/s [*input_shape,nt]

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
    @shapeDecorator
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

           V(t) in km/s [*input_shape,nt]

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
    @shapeDecorator
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

           W(t) in km/s [*input_shape,nt]

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

    @shapeDecorator
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

           SkyCoord(t) [*input_shape,nt] 

        HISTORY:

           2019-02-21 - Written - Bovy (UofT)

        """
        kwargs.pop('quantity',None) # rm useless keyword to no conflict later
        kwargs['dontreshape']= True
        _check_roSet(self,kwargs,'SkyCoord')
        thiso= self._call_internal(*args,**kwargs)
        thiso_shape= thiso.shape
        thiso= thiso.reshape((thiso_shape[0],-1))
        radec= _radec(self,thiso,*args,**kwargs).T\
            .reshape((2,)+thiso_shape[1:])
        tdist= self.dist(quantity=False,*args,**kwargs).T
        if not _APY3: # pragma: no cover
            kwargs.pop('dontreshape')
            return coordinates.SkyCoord(radec[0]*units.degree,
                                        radec[1]*units.degree,
                                        distance=tdist*units.kpc,
                                        frame='icrs').T
        _check_voSet(self,kwargs,'SkyCoord')
        pmrapmdec= _pmrapmdec(self,thiso,*args,**kwargs).T\
            .reshape((2,)+thiso_shape[1:])
        tvlos= self.vlos(quantity=False,*args,**kwargs).T
        kwargs.pop('dontreshape')
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


    def __call__(self,*args,**kwargs):
        """
        NAME:
 
          __call__

        PURPOSE:

           return the orbits at time t

        INPUT:

           t - desired time (can be Quantity)

        OUTPUT:

           an Orbit instance with initial conditions set to the 
           phase-space at time t; shape of new Orbit is (shape_old,nt)

        HISTORY:

           2019-03-05 - Written - Bovy (UofT)

           2019-03-20 - Implemented multiple times --> Orbits - Bovy (UofT)

        """
        orbSetupKwargs= {'ro':self._ro,
                         'vo':self._vo,
                         'zo':self._zo,
                         'solarmotion':self._solarmotion}
        thiso= self._call_internal(*args,**kwargs)
        out= Orbit(vxvv=numpy.reshape(thiso.T,self.shape+thiso.T.shape[1:]),
                   **orbSetupKwargs)
        out._roSet= self._roSet
        out._voSet= self._voSet
        return out

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
        if len(args) == 0 or (not hasattr(self,'t') and args[0] == 0. ):
            return numpy.array(self.vxvv).T
        elif not hasattr(self,'t'):
            raise ValueError("Integrate instance before evaluating it at a specific time")
        else:
            t= args[0]
        # Parse t, first check whether we are dealing with the common case 
        # where one wants all integrated times
        # 2nd line: scalar Quantities have __len__, but raise TypeError 
        # for scalars
        t_exact_integration_times= hasattr(t,'__len__') \
            and not (_APY_LOADED and isinstance(t,units.Quantity) 
                     and t.isscalar) \
            and (len(t) == len(self.t)) \
            and numpy.all(t == self.t)
        if _APY_LOADED and isinstance(t,units.Quantity):
            t= conversion.parse_time(t,ro=self._ro,vo=self._vo)
            # Need to re-evaluate now that t has changed...
            t_exact_integration_times= hasattr(t,'__len__') \
                and (len(t) == len(self.t)) \
                and numpy.all(t == self.t)
        elif '_integrate_t_asQuantity' in self.__dict__ \
                and self._integrate_t_asQuantity \
                and not t_exact_integration_times:
            # Not doing hasattr in above elif, bc currently slow due to overwrite of __getattribute__
            warnings.warn("You specified integration times as a Quantity, but are evaluating at times not specified as a Quantity; assuming that time given is in natural (internal) units (multiply time by unit to get output at physical time)",galpyWarning)
        if t_exact_integration_times: # Common case where one wants all integrated times
            return self.orbit.T.copy()
        elif isinstance(t,(int,float,numpy.number)) and hasattr(self,'t') \
                and t in list(self.t):
            return numpy.array(self.orbit[:,list(self.t).index(t),:]).T
        else:
            if isinstance(t,(int,float,numpy.number)): 
                nt= 1
                t= numpy.atleast_1d(t)
            else: 
                nt= len(t)
            if numpy.any(t > numpy.nanmax(self.t)) \
                    or numpy.any(t < numpy.nanmin(self.t)):
                raise ValueError('Found time value not in the integration time domain')
            try:
                self._setupOrbitInterp()
            except:
                out= numpy.zeros((self.phasedim(),nt,self.size))
                for jj in range(nt):
                    try:
                        indx= list(self.t).index(t[jj])
                    except ValueError:
                        raise LookupError("Orbit interpolaton failed; integrate on finer grid")
                    out[:,jj]= self.orbit[:,indx].T
                return out #should always have nt > 1, bc otherwise covered by above
            out= numpy.empty((self.phasedim(),nt,self.size))
            # Evaluating RectBivariateSpline on grid requires sorted arrays
            sindx= numpy.argsort(t)
            t= t[sindx]
            usindx= numpy.argsort(sindx) # to later unsort
            if self.phasedim() == 4 or self.phasedim() == 6:
                #Unpack interpolated x and y to R and phi
                x= self._orbInterp[0](t,self._orb_indx_4orbInterp)
                y= self._orbInterp[-1](t,self._orb_indx_4orbInterp)
                out[0]= numpy.sqrt(x*x+y*y)
                out[-1]= numpy.arctan2(y,x)
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

    def toPlanar(self):
        """
        NAME:

           toPlanar

        PURPOSE:

           convert 3D orbits into 2D orbits

        INPUT:

           (none)

        OUTPUT:

           planar Orbit instance

        HISTORY:

           2019-03-02 - Written - Bovy (UofT)

        """
        orbSetupKwargs= {'ro':self._ro,
                         'vo':self._vo,
                         'zo':self._zo,
                         'solarmotion':self._solarmotion}
        if self.phasedim() == 6:
            vxvv= self.vxvv[:,[0,1,2,5]]
        elif self.phasedim() == 5:
            vxvv= self.vxvv[:,[0,1,2]]
        else:
            raise AttributeError("planar or linear Orbit does not have the toPlanar attribute")
        out= Orbit(vxvv=vxvv,**orbSetupKwargs)
        out._roSet= self._roSet
        out._voSet= self._voSet
        return out

    def toLinear(self):
        """
        NAME:

           toLinear

        PURPOSE:

           convert 3D orbits into 1D orbits (z)

        INPUT:

           (none)

        OUTPUT:

           linear Orbit instance

        HISTORY:

           2019-03-02 - Written - Bovy (UofT)

        """
        orbSetupKwargs= {'ro':self._ro,
                         'vo':self._vo}
        if self.dim() == 3:
            vxvv= self.vxvv[:,[3,4]]
        else:
            raise AttributeError("planar or linear Orbit does not have the toPlanar attribute")
        out= Orbit(vxvv=vxvv,**orbSetupKwargs)
        out._roSet= self._roSet
        out._voSet= self._voSet
        return out

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
        orb_indx= numpy.arange(self.size)
        for ii in range(self.phasedim()):
            if (self.phasedim() == 4 or self.phasedim() == 6) and ii == 0:
                #Interpolate x and y rather than R and phi to avoid issues w/ phase wrapping
                if self.size == 1:
                    orbInterp.append(_1DInterp(\
                          self.t,
                          self.orbit[0,:,0]*numpy.cos(self.orbit[0,:,-1])))
                else:
                    orbInterp.append(interpolate.RectBivariateSpline(\
                          self.t,orb_indx,
                          (self.orbit[:,:,0]*numpy.cos(self.orbit[:,:,-1])).T,
                          ky=1,s=0.))
            elif (self.phasedim() == 4 or self.phasedim() == 6) and \
                    ii == self.phasedim()-1:
                if self.size == 1:
                    orbInterp.append(_1DInterp(\
                          self.t,
                          self.orbit[0,:,0]*numpy.sin(self.orbit[0,:,-1])))
                else:
                    orbInterp.append(interpolate.RectBivariateSpline(\
                          self.t,orb_indx,
                          (self.orbit[:,:,0]*numpy.sin(self.orbit[:,:,-1])).T,
                          ky=1,s=0.))
            else:
                if self.size == 1:
                    orbInterp.append(_1DInterp(self.t,self.orbit[0,:,ii]))
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

    def _parse_plot_quantity(self,quant,**kwargs):
        """Internal function to parse a quantity to be plotted based on input data"""
        # Cannot be using Quantity output
        kwargs['quantity']= False
        if callable(quant):
            out= quant(self.t)
            if out.shape == self.t.shape:
                out= numpy.tile(out,(len(self.vxvv),1))
            return out
        def _eval(q):
            # Check those that don't have the exact name of the function
            if q == 't':
                # Typically expect this to have same shape as other quantities
                return numpy.tile(self.time(self.t,**kwargs),
                                  (len(self.vxvv),1))
            elif q == 'Enorm':
                return (self.E(self.t,**kwargs).T/self.E(0.,**kwargs)).T
            elif q == 'Eznorm':
                return (self.Ez(self.t,**kwargs).T/self.Ez(0.,**kwargs)).T
            elif q == 'ERnorm':
                return (self.ER(self.t,**kwargs).T/self.ER(0.,**kwargs)).T
            elif q == 'Jacobinorm':
                return (self.Jacobi(self.t,**kwargs).T/self.Jacobi(0.,**kwargs)).T
            else: # these are exact, e.g., 'x' for self.x
                return self.__getattribute__(q)(self.t,**kwargs)
        try:
            return _eval(quant)
        except AttributeError: pass
        try:
            import numexpr
        except ImportError: #pragma: no cover
            raise ImportError('Parsing the quantity to be plotted failed; if you are trying to plot an expression, please make sure to install numexpr first')
        # Figure out the variables in the expression to be computed to plot
        try:
            vars= numexpr.NumExpr(quant).input_names
        except TypeError as err:
            raise TypeError('Parsing the expression {} failed, with error message:\n"{}"'.format(quant,err))
        # Construct dictionary of necessary parameters
        vars_dict= {}
        for var in vars:
            vars_dict[var]= _eval(var)
        return numexpr.evaluate(quant,local_dict=vars_dict)

    def plot(self,*args,**kwargs):
        """
        NAME:

           plot

        PURPOSE:

           plot a previously calculated orbit (with reasonable defaults)

        INPUT:

           d1= first dimension to plot ('x', 'y', 'R', 'vR', 'vT', 'z', 'vz', ...); can also be an expression, like 'R*vR', or a  user-defined function of time (e.g., lambda t: o.R(t) for R)

           d2= second dimension to plot; can also be an expression, like 'R*vR', or a user-defined function of time (e.g., lambda t: o.R(t) for R)

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           matplotlib.plot inputs+galpy.util.plot.plot inputs

        OUTPUT:

           sends plot to output device

        HISTORY:

           2010-07-26 - Written - Bovy (NYU)

           2010-09-22 - Adapted to more general framework - Bovy (NYU)

           2013-11-29 - added ra,dec kwargs and other derived quantities - Bovy (IAS)

           2014-06-11 - Support for plotting in physical coordinates - Bovy (IAS)

           2017-11-28 - Allow arbitrary functions of time to be plotted - Bovy (UofT)

           2019-04-13 - Edited for multiple Orbits - Bovy (UofT)

        """
        if (kwargs.get('use_physical',False) \
                and kwargs.get('ro',self._roSet)) or \
                (not 'use_physical' in kwargs \
                     and kwargs.get('ro',self._roSet)):
            labeldict= {'t':r'$t\ (\mathrm{Gyr})$','R':r'$R\ (\mathrm{kpc})$',
                        'vR':r'$v_R\ (\mathrm{km\,s}^{-1})$',
                        'vT':r'$v_T\ (\mathrm{km\,s}^{-1})$',
                        'z':r'$z\ (\mathrm{kpc})$',
                        'vz':r'$v_z\ (\mathrm{km\,s}^{-1})$','phi':r'$\phi$',
                        'r':r'$r\ (\mathrm{kpc})$',
                        'x':r'$x\ (\mathrm{kpc})$','y':r'$y\ (\mathrm{kpc})$',
                        'vx':r'$v_x\ (\mathrm{km\,s}^{-1})$',
                        'vy':r'$v_y\ (\mathrm{km\,s}^{-1})$',
                        'E':r'$E\,(\mathrm{km}^2\,\mathrm{s}^{-2})$',
                        'Ez':r'$E_z\,(\mathrm{km}^2\,\mathrm{s}^{-2})$',
                        'ER':r'$E_R\,(\mathrm{km}^2\,\mathrm{s}^{-2})$',
                        'Enorm':r'$E(t)/E(0.)$',
                        'Eznorm':r'$E_z(t)/E_z(0.)$',
                        'ERnorm':r'$E_R(t)/E_R(0.)$',
                        'Jacobi':r'$E-\Omega_p\,L\,(\mathrm{km}^2\,\mathrm{s}^{-2})$',
                        'Jacobinorm':r'$(E-\Omega_p\,L)(t)/(E-\Omega_p\,L)(0)$'}
        else:
            labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                        'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                        'r':r'$r$',
                        'x':r'$x$','y':r'$y$','vx':r'$v_x$','vy':r'$v_y$',
                        'E':r'$E$','Enorm':r'$E(t)/E(0.)$',
                        'Ez':r'$E_z$','Eznorm':r'$E_z(t)/E_z(0.)$',
                        'ER':r'$E_R$','ERnorm':r'$E_R(t)/E_R(0.)$',
                        'Jacobi':r'$E-\Omega_p\,L$',
                        'Jacobinorm':r'$(E-\Omega_p\,L)(t)/(E-\Omega_p\,L)(0)$'}
        labeldict.update({'ra':r'$\alpha\ (\mathrm{deg})$',
                          'dec':r'$\delta\ (\mathrm{deg})$',
                          'll':r'$l\ (\mathrm{deg})$',
                          'bb':r'$b\ (\mathrm{deg})$',
                          'dist':r'$d\ (\mathrm{kpc})$',
                          'pmra':r'$\mu_\alpha\ (\mathrm{mas\,yr}^{-1})$',
                          'pmdec':r'$\mu_\delta\ (\mathrm{mas\,yr}^{-1})$',
                          'pmll':r'$\mu_l\ (\mathrm{mas\,yr}^{-1})$',
                          'pmbb':r'$\mu_b\ (\mathrm{mas\,yr}^{-1})$',
                          'vlos':r'$v_\mathrm{los}\ (\mathrm{km\,s}^{-1})$',
                          'helioX':r'$X\ (\mathrm{kpc})$',
                          'helioY':r'$Y\ (\mathrm{kpc})$',
                          'helioZ':r'$Z\ (\mathrm{kpc})$',
                          'U':r'$U\ (\mathrm{km\,s}^{-1})$',
                          'V':r'$V\ (\mathrm{km\,s}^{-1})$',
                          'W':r'$W\ (\mathrm{km\,s}^{-1})$'})
        # Cannot be using Quantity output
        kwargs['quantity']= False
        #Defaults
        if not 'd1' in kwargs and not 'd2' in kwargs:
            if self.phasedim() == 3:
                d1= 'R'
                d2= 'vR'
            elif self.phasedim() == 4:
                d1= 'x'
                d2= 'y'
            elif self.phasedim() == 2:
                d1= 'x'
                d2= 'vx'
            elif self.phasedim() == 5 or self.phasedim() == 6:
                d1= 'R'
                d2= 'z'
        elif not 'd1' in kwargs:
            d2=  kwargs.pop('d2')
            d1= 't'
        elif not 'd2' in kwargs:
            d1= kwargs.pop('d1')
            d2= 't'
        else:
            d1= kwargs.pop('d1')
            d2= kwargs.pop('d2')
        kwargs['dontreshape']= True
        x= numpy.atleast_2d(self._parse_plot_quantity(d1,**kwargs))
        y= numpy.atleast_2d(self._parse_plot_quantity(d2,**kwargs))
        kwargs.pop('dontreshape')
        kwargs.pop('ro',None)
        kwargs.pop('vo',None)
        kwargs.pop('obs',None)
        kwargs.pop('use_physical',None)
        kwargs.pop('pot',None)
        kwargs.pop('OmegaP',None)
        kwargs.pop('quantity',None)
        auto_scale= not 'xrange' in kwargs and not 'yrange' in kwargs \
            and not kwargs.get('overplot',False)
        labels= kwargs.pop('label',['Orbit {}'.format(ii+1) 
                                    for ii in range(self.size)])
        if self.size == 1 and isinstance(labels,str): labels= [labels]
        #Plot
        if not 'xlabel' in kwargs:
            kwargs['xlabel']= labeldict.get(d1,r'${}$'.format(d1))
        if not 'ylabel' in kwargs:
            kwargs['ylabel']= labeldict.get(d2,r'${}$'.format(d2))
        for ii,(tx,ty) in enumerate(zip(x,y)):
            kwargs['label']= labels[ii]
            line2d= plot.plot(tx,ty,*args,**kwargs)[0]
            kwargs['overplot']= True
        if auto_scale: line2d.axes.autoscale(enable=True)
        plot._add_ticks()
        return [line2d]

    def plot3d(self,*args,**kwargs):
        """
        NAME:

           plot3d

        PURPOSE:

           plot 3D aspects of an Orbit

        INPUT:

           d1= first dimension to plot ('x', 'y', 'R', 'vR', 'vT', 'z', 'vz', ...); can also be an expression, like 'R*vR', or a user-defined function of time (e.g., lambda t: o.R(t) for R)

           d2= second dimension to plot

           d3= third dimension to plot

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

           galpy.util.plot.plot3d args and kwargs

        OUTPUT:

           plot

        HISTORY:

           2010-07-26 - Written - Bovy (NYU)

           2010-09-22 - Adapted to more general framework - Bovy (NYU)

           2010-01-08 - Adapted to 3D - Bovy (NYU)

           2013-11-29 - added ra,dec kwargs and other derived quantities - Bovy (IAS)

           2014-06-11 - Support for plotting in physical coordinates - Bovy (IAS)

           2017-11-28 - Allow arbitrary functions of time to be plotted - Bovy (UofT)

           2019-04-13 - Adapated for multiple orbits - Bovy (UofT)

        """
        if (kwargs.get('use_physical',False) \
                and kwargs.get('ro',self._roSet)) or \
                (not 'use_physical' in kwargs \
                     and kwargs.get('ro',self._roSet)):
            labeldict= {'t':r'$t\ (\mathrm{Gyr})$','R':r'$R\ (\mathrm{kpc})$',
                        'vR':r'$v_R\ (\mathrm{km\,s}^{-1})$',
                        'vT':r'$v_T\ (\mathrm{km\,s}^{-1})$',
                        'z':r'$z\ (\mathrm{kpc})$',
                        'vz':r'$v_z\ (\mathrm{km\,s}^{-1})$','phi':r'$\phi$',
                        'r':r'$r\ (\mathrm{kpc})$',
                        'x':r'$x\ (\mathrm{kpc})$','y':r'$y\ (\mathrm{kpc})$',
                        'vx':r'$v_x\ (\mathrm{km\,s}^{-1})$',
                        'vy':r'$v_y\ (\mathrm{km\,s}^{-1})$'}
        else:
            labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                        'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                        'r':r'$r$','x':r'$x$','y':r'$y$',
                        'vx':r'$v_x$','vy':r'$v_y$'}
        labeldict.update({'ra':r'$\alpha\ (\mathrm{deg})$',
                          'dec':r'$\delta\ (\mathrm{deg})$',
                          'll':r'$l\ (\mathrm{deg})$',
                          'bb':r'$b\ (\mathrm{deg})$',
                          'dist':r'$d\ (\mathrm{kpc})$',
                          'pmra':r'$\mu_\alpha\ (\mathrm{mas\,yr}^{-1})$',
                          'pmdec':r'$\mu_\delta\ (\mathrm{mas\,yr}^{-1})$',
                          'pmll':r'$\mu_l\ (\mathrm{mas\,yr}^{-1})$',
                          'pmbb':r'$\mu_b\ (\mathrm{mas\,yr}^{-1})$',
                          'vlos':r'$v_\mathrm{los}\ (\mathrm{km\,s}^{-1})$',
                          'helioX':r'$X\ (\mathrm{kpc})$',
                          'helioY':r'$Y\ (\mathrm{kpc})$',
                          'helioZ':r'$Z\ (\mathrm{kpc})$',
                          'U':r'$U\ (\mathrm{km\,s}^{-1})$',
                          'V':r'$V\ (\mathrm{km\,s}^{-1})$',
                          'W':r'$W\ (\mathrm{km\,s}^{-1})$'})
        # Cannot be using Quantity output
        kwargs['quantity']= False
        #Defaults
        if not 'd1' in kwargs and not 'd2' in kwargs and not 'd3' in kwargs:
            if self.phasedim() == 3:
                d1= 'R'
                d2= 'vR'
                d3= 'vT'
            elif self.phasedim() == 4:
                d1= 'x'
                d2= 'y'
                d3= 'vR'
            elif self.phasedim() == 2:
                raise AttributeError("Cannot plot 3D aspects of 1D orbits")
            elif self.phasedim() == 5:
                d1= 'R'
                d2= 'vR'
                d3= 'z'
            elif self.phasedim() == 6:
                d1= 'x'
                d2= 'y'
                d3= 'z'
        elif not ('d1' in kwargs and 'd2' in kwargs and 'd3' in kwargs):
            raise AttributeError("Please provide 'd1', 'd2', and 'd3'")
        else:
            d1= kwargs.pop('d1')
            d2= kwargs.pop('d2')
            d3= kwargs.pop('d3')
        kwargs['dontreshape']= True
        x= numpy.atleast_2d(self._parse_plot_quantity(d1,**kwargs))
        y= numpy.atleast_2d(self._parse_plot_quantity(d2,**kwargs))
        z= numpy.atleast_2d(self._parse_plot_quantity(d3,**kwargs))
        kwargs.pop('dontreshape')
        kwargs.pop('ro',None)
        kwargs.pop('vo',None)
        kwargs.pop('obs',None)
        kwargs.pop('use_physical',None)
        kwargs.pop('quantity',None)
        auto_scale= not 'xrange' in kwargs and not 'yrange' in kwargs \
            and not 'zrange' in kwargs and not kwargs.get('overplot',False)
        #Plot
        if not 'xlabel' in kwargs:
            kwargs['xlabel']= labeldict.get(d1,r'${}$'.format(d1))
        if not 'ylabel' in kwargs:
            kwargs['ylabel']= labeldict.get(d2,r'${}$'.format(d2))
        if not 'zlabel' in kwargs:
            kwargs['zlabel']= labeldict.get(d3,r'${}$'.format(d3))
        for tx,ty,tz in zip(x,y,z):
            line3d= plot.plot3d(tx,ty,tz,*args,**kwargs)[0]
            kwargs['overplot']= True
        if auto_scale: line3d.axes.autoscale(enable=True)
        plot._add_ticks()
        return [line3d]

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

           staticPlot= (False) if True, create a static plot that doesn't allow zooming, panning, etc.

           ro= (Object-wide default) physical scale for distances to use to convert (can be Quantity)

           vo= (Object-wide default) physical scale for velocities to use to convert (can be Quantity)

           use_physical= use to override Object-wide default for using a physical scale for output

        OUTPUT:

           IPython.display.HTML object with code to animate the orbit; can be directly shown in jupyter notebook or embedded in HTML pages; get a text version of the HTML using the _repr_html_() function

        HISTORY:

           2017-09-17-24 - Written - Bovy (UofT)

           2019-03-11 - Adapted for multiple orbits - Bovy (UofT)

        """
        try:
            from IPython.display import HTML
        except ImportError:
            raise ImportError("Orbit.animate requires ipython/jupyter to be installed")
        if (kwargs.get('use_physical',False) \
                and kwargs.get('ro',self._roSet)) or \
                (not 'use_physical' in kwargs \
                     and kwargs.get('ro',self._roSet)):
            labeldict= {'t':'t (Gyr)',
                        'R':'R (kpc)',
                        'vR':'v_R (km/s)',
                        'vT':'v_T (km/s)',
                        'z':'z (kpc)',
                        'vz':'v_z (km/s)',
                        'phi':'azimuthal angle',
                        'r':'r (kpc)',
                        'x':'x (kpc)',
                        'y':'y (kpc)',
                        'vx':'v_x (km/s)',
                        'vy':'v_y (km/s)',
                        'E':'E (km^2/s^2)',
                        'Ez':'E_z (km^2/s^2)',
                        'ER':'E_R (km^2/s^2)',
                        'Enorm':'E(t)/E(0.)',
                        'Eznorm':'E_z(t)/E_z(0.)',
                        'ERnorm':'E_R(t)/E_R(0.)',
                        'Jacobi':'E-Omega_p L (km^2/s^2)',
                        'Jacobinorm':'(E-Omega_p L)(t)/(E-Omega_p L)(0)'}
        else:
            labeldict= {'t':'t','R':'R','vR':'v_R','vT':'v_T',
                        'z':'z','vz':'v_z','phi':r'azimuthal angle',
                        'r':'r',
                        'x':'x','y':'y','vx':'v_x','vy':'v_y',
                        'E':'E','Enorm':'E(t)/E(0.)',
                        'Ez':'E_z','Eznorm':'E_z(t)/E_z(0.)',
                        'ER':r'E_R','ERnorm':r'E_R(t)/E_R(0.)',
                        'Jacobi':r'E-Omega_p L',
                        'Jacobinorm':r'(E-Omega_p L)(t)/(E-Omega_p L)(0)'}
        labeldict.update({'ra':'RA (deg)',
                          'dec':'Dec (deg)',
                          'll':'Galactic lon (deg)',
                          'bb':'Galactic lat (deg)',
                          'dist':'distance (kpc)',
                          'pmra':'pmRA (mas/yr)',
                          'pmdec':'pmDec (mas/yr)',
                          'pmll':'pmGlon (mas/yr)',
                          'pmbb':'pmGlat (mas/yr)',
                          'vlos':'line-of-sight vel (km/s)',
                          'helioX':'X (kpc)',
                          'helioY':'Y (kpc)',
                          'helioZ':'Z (kpc)',
                          'U':'U (km/s)',
                          'V':'V (km/s)',
                          'W':'W (km/s)'})
        # Cannot be using Quantity output
        kwargs['quantity']= False
        #Defaults
        if not 'd1' in kwargs and not 'd2' in kwargs:
            if self.phasedim() == 3:
                d1= 'R'
                d2= 'vR'
            elif self.phasedim() == 4:
                d1= 'x'
                d2= 'y'
            elif self.phasedim() == 2:
                d1= 'x'
                d2= 'vx'
            elif self.dim() == 3:
                d1= 'R'
                d2= 'z'
        elif not 'd1' in kwargs:
            d2=  kwargs.pop('d2')
            d1= 't'
        elif not 'd2' in kwargs:
            d1= kwargs.pop('d1')
            d2= 't'
        else:
            d1= kwargs.pop('d1')
            d2= kwargs.pop('d2')
        xs= []
        ys= []
        xlabels= []
        ylabels= []
        if isinstance(d1,str) or callable(d1):
            d1s= [d1]
            d2s= [d2]
        else:
            d1s= d1
            d2s= d2
        if len(d1s) > 3:
            raise ValueError('Orbit.animate only works for up to three subplots')
        all_xlabel= kwargs.get('xlabel',[None for d in d1])
        all_ylabel= kwargs.get('ylabel',[None for d in d2])
        for d1,d2, xlabel, ylabel in zip(d1s,d2s,all_xlabel,all_ylabel):
           #Get x and y for each subplot
            kwargs['dontreshape']= True
            x= self._parse_plot_quantity(d1,**kwargs)
            y= self._parse_plot_quantity(d2,**kwargs)
            kwargs.pop('dontreshape')
            xs.append(x)
            ys.append(y)
            if xlabel is None:
                xlabels.append(labeldict.get(d1,'\mathrm{No\ xlabel\ specified}'))
            else:
                xlabels.append(xlabel)
            if ylabel is None:
                ylabels.append(labeldict.get(d2,'\mathrm{No\ ylabel\ specified}'))
            else:
                ylabels.append(ylabel)
        kwargs.pop('ro',None)
        kwargs.pop('vo',None)
        kwargs.pop('obs',None)
        kwargs.pop('use_physical',None)
        kwargs.pop('pot',None)
        kwargs.pop('OmegaP',None)
        kwargs.pop('quantity',None)
        width= kwargs.pop('width',600)
        height= kwargs.pop('height',400)
        load_jslibs= kwargs.pop('load_jslibs',True)
        if load_jslibs:
            load_jslibs_code= """
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.5/require.min.js"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
"""
        else:
            load_jslibs_code= ""
        # Dump data to HTML
        nplots= len(xs)
        jsonDict= {}
        for ii in range(nplots):
            for jj in range(self.size):
                jsonDict['x%i_%i' % (ii+1,jj)]= xs[ii][jj].tolist()
                jsonDict['y%i_%i' % (ii+1,jj)]= ys[ii][jj].tolist()
        json_filename= kwargs.pop('json_filename',None)
        if json_filename is None:
            jd= json.dumps(jsonDict)
            json_code= """  let data= JSON.parse('{jd}');""".format(jd=jd)
            close_json_code= ""
        else:
            with open(json_filename,'w') as jfile:
                json.dump(jsonDict,jfile)
            json_code= """Plotly.d3.json('{jfilename}',function(data){{""".format(jfilename=json_filename)
            close_json_code= "});"
        self.divid= 'galpy-'\
            +''.join(choice(ascii_lowercase) for i in range(24))
        button_width= 419.51+4.*10.
        button_margin_left= int(numpy.round((width-button_width)/2.))
        if button_margin_left < 0: button_margin_left= 0
        # Configuration options
        config= """{{staticPlot: {staticPlot}}}"""\
            .format(staticPlot='true' if kwargs.pop('staticPlot',False)
                    else 'false')
        # Layout for multiple plots
        if len(d1s) == 1:
            xmin= [0,0,0]
            xmax= [1,1,1]
        elif len(d1s) == 2:
            xmin= [0,0.55,0]
            xmax= [0.45,1,1]
        elif len(d1s) == 3:
            xmin= [0,0.365,0.73]
            xmax= [0.27,0.635,1]
        # Colors
        line_colors= ['#1f77b4', # muted blue
                      '#ff7f0e', # safety orange
                      '#2ca02c', # cooked asparagus green
                      '#d62728', # brick red
                      '#9467bd', # muted purple
                      '#8c564b', # chestnut brown
                      '#e377c2', # raspberry yogurt pink
                      '#7f7f7f', # middle gray
                      '#bcbd22', # curry yellow-green
                      '#17becf'] # blue-teal
        # When there are more than these # of colors needed, make up randoms
        if self.size > len(line_colors):
            line_colors.extend(["#%06x" % numpy.random.randint(0, 0xFFFFFF)
                                for ii in range(self.size-len(line_colors))])
        layout= """{{
  xaxis: {{
    title: '{xlabel}',
    domain: [{xmin},{xmax}],
}},
  yaxis: {{title: '{ylabel}'}},
  margin: {{t: 20}},
  hovermode: 'closest',
  showlegend: false,
""".format(xlabel=xlabels[0],ylabel=ylabels[0],xmin=xmin[0],xmax=xmax[0])

        for ii in range(1,nplots):
            layout+= """  xaxis{idx}: {{
    title: '{xlabel}',
    anchor: 'y{idx}',
    domain: [{xmin},{xmax}],
}},
  yaxis{idx}: {{
    title: '{ylabel}',
    anchor: 'x{idx}',
}},
""".format(idx=ii+1,xlabel=xlabels[ii],ylabel=ylabels[ii],
           xmin=xmin[ii],xmax=xmax[ii])
        layout+="""}"""
        # First plot
        setup_trace1= """
    let trace1= {{
      x: data.x1_0.slice(0,numPerFrame), 
      y: data.y1_0.slice(0,numPerFrame),
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 0.8,
        color: '{line_color}',
       }},
    }};

    let trace2= {{
      x: data.x1_0.slice(0,numPerFrame), 
      y: data.y1_0.slice(0,numPerFrame),
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 3.,
        color: '{line_color}',
        }},
    }};
""".format(line_color=line_colors[0])
        traces_cumul= """trace1,trace2"""
        for ii in range(1,self.size):
            setup_trace1+= """            
    let trace{trace_num_1}= {{
      x: data.x1_{trace_indx}.slice(0,numPerFrame), 
      y: data.y1_{trace_indx}.slice(0,numPerFrame),
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 0.8,
        color: '{line_color}',
       }},
    }};

    let trace{trace_num_2}= {{
      x: data.x1_{trace_indx}.slice(0,numPerFrame), 
      y: data.y1_{trace_indx}.slice(0,numPerFrame),
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 3.,
        color: '{line_color}',
        }},
    }};
""".format(trace_indx=str(ii),trace_num_1=str(2*ii+1),trace_num_2=str(2*ii+2),
           line_color=line_colors[ii])
            traces_cumul+= """,trace{trace_num_1},trace{trace_num_2}""".format(trace_num_1=str(2*ii+1),trace_num_2=str(2*ii+2))
        x_data_list = """"""
        y_data_list = """"""
        trace_num_10_list = """"""
        trace_num_20_list = """"""
        for jj in range(len(d1s)):
            for ii in range(0, self.size):
                x_data_list += """data.x{jj}_{trace_indx}.slice(trace_slice_begin,trace_slice_end), """.format(jj=jj+1,
                    divid=self.divid, trace_indx=str(ii))
                y_data_list += """data.y{jj}_{trace_indx}.slice(trace_slice_begin,trace_slice_end), """.format(jj=jj+1,
                    divid=self.divid, trace_indx=str(ii))
                trace_num_10_list += """{trace_num_10}, """.format(trace_num_10=str(2*jj*self.size + 2 * ii + 1 - 1))
                trace_num_20_list += """{trace_num_20}, """.format(trace_num_20=str(2*jj*self.size + 2 * ii + 2 - 1))
        # Additional traces for additional plots
        if len(d1s) > 1:
            setup_trace2= """
    let trace{trace_num_1}= {{
      x: data.x2_0.slice(0,numPerFrame),
      y: data.y2_0.slice(0,numPerFrame),
      xaxis: 'x2',
      yaxis: 'y2',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 0.8,
        color: '{line_color}',
      }},
    }};

    let trace{trace_num_2}= {{
      x: data.x2_0.slice(0,numPerFrame), 
      y: data.y2_0.slice(0,numPerFrame),
      xaxis: 'x2',
      yaxis: 'y2',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 3.,
        color: '{line_color}',
      }},
    }};
""".format(line_color=line_colors[0],trace_num_1=str(2*self.size+1),
           trace_num_2=str(2*self.size+2))
            traces_cumul+= """,trace{trace_num_1},trace{trace_num_2}""".format(trace_num_1=str(2*self.size+1),trace_num_2=str(2*self.size+2))
            for ii in range(1,self.size):
                setup_trace2+= """
    let trace{trace_num_1}= {{
      x: data.x2_{trace_indx}.slice(0,numPerFrame),
      y: data.y2_{trace_indx}.slice(0,numPerFrame),
      xaxis: 'x2',
      yaxis: 'y2',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 0.8,
        color: '{line_color}',
      }},
    }};

    let trace{trace_num_2}= {{
      x: data.x2_{trace_indx}.slice(0,numPerFrame), 
      y: data.y2_{trace_indx}.slice(0,numPerFrame),
      xaxis: 'x2',
      yaxis: 'y2',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 3.,
        color: '{line_color}',
      }},
    }};
""".format(line_color=line_colors[ii],trace_indx=str(ii),
           trace_num_1=str(2*self.size+2*ii+1),
           trace_num_2=str(2*self.size+2*ii+2))
                traces_cumul+= """,trace{trace_num_1},trace{trace_num_2}""".format(trace_num_1=str(2*self.size+2*ii+1),trace_num_2=str(2*self.size+2*ii+2))
        else: # else for "if there is a 2nd panel"
            setup_trace2= """
    let traces= [{traces_cumul}];
""".format(traces_cumul=traces_cumul)
        if len(d1s) > 2:
            setup_trace3= """
    let trace{trace_num_1}= {{
      x: data.x3_0.slice(0,numPerFrame),
      y: data.y3_0.slice(0,numPerFrame),
      xaxis: 'x3',
      yaxis: 'y3',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 0.8,
        color: '{line_color}',
      }},
    }};

    let trace{trace_num_2}= {{
      x: data.x3_0.slice(0,numPerFrame), 
      y: data.y3_0.slice(0,numPerFrame),
      xaxis: 'x3',
      yaxis: 'y3',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 3.,
        color: '{line_color}',
      }},
    }};
""".format(line_color=line_colors[0],trace_num_1=str(4*self.size+1),
           trace_num_2=str(4*self.size+2))
            traces_cumul+= """,trace{trace_num_1},trace{trace_num_2}""".format(trace_num_1=str(4*self.size+1),trace_num_2=str(4*self.size+2))
            for ii in range(1,self.size):
                setup_trace3+= """
    let trace{trace_num_1}= {{
      x: data.x3_{trace_indx}.slice(0,numPerFrame),
      y: data.y3_{trace_indx}.slice(0,numPerFrame),
      xaxis: 'x3',
      yaxis: 'y3',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 0.8,
        color: '{line_color}',
      }},
    }};

    let trace{trace_num_2}= {{
      x: data.x3_{trace_indx}.slice(0,numPerFrame), 
      y: data.y3_{trace_indx}.slice(0,numPerFrame),
      xaxis: 'x3',
      yaxis: 'y3',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 3.,
        color: '{line_color}',
      }},
    }};
""".format(line_color=line_colors[ii],trace_indx=str(ii),
           trace_num_1=str(4*self.size+2*ii+1),
           trace_num_2=str(4*self.size+2*ii+2),
           trace_num_10=str(4*self.size+2*ii+1-1),
           trace_num_20=str(4*self.size+2*ii+2-1))
                traces_cumul+= """,trace{trace_num_1},trace{trace_num_2}""".format(trace_num_1=str(4*self.size+2*ii+1),trace_num_2=str(4*self.size+2*ii+2))
            setup_trace3 += """
            let traces= [{traces_cumul}];
            """.format(traces_cumul=traces_cumul)
        elif len(d1s) > 1: # elif for "if there is a 3rd panel
            setup_trace3= """
    let traces= [{traces_cumul}];
""".format(traces_cumul=traces_cumul)
        else: # else for "if there is a 3rd or 2nd panel" (don't think we can get here!)
            setup_trace3= ""
        return HTML("""
<style>
.galpybutton {{
    background-color:#ffffff;
    -moz-border-radius:16px;
    -webkit-border-radius:16px;
    border-radius:16px;
    border:1px solid #1f77b4;
    display:inline-block;
    cursor:pointer;
    color:#1f77b4;
    font-family:Courier;
    font-size:17px;
    padding:8px 10px;
    text-decoration:none;
    text-shadow:0px 1px 0px #2f6627;
}}
.galpybutton:hover {{
    background-color:#ffffff;
}}
.galpybutton:active {{
    position:relative;
    top:1px;
}}
.galpybutton:focus{{
    outline:0;
}}
</style>

<div id='{divid}' style='width:{width}px;height:{height}px;'></div>
<div class="controlbutton" id="{divid}-play" style="margin-left:{button_margin_left}px;display: inline-block;">
<button class="galpybutton">Play</button></div>
<div class="controlbutton" id="{divid}-pause" style="margin-left:10px;display: inline-block;">
<button class="galpybutton">Pause</button></div>
<div class="controlbutton" id="{divid}-timestwo" style="margin-left:10px;display: inline-block;">
<button class="galpybutton">Speed<font face="Arial">&thinsp;</font>x<font face="Arial">&thinsp;</font>2</button></div>
<div class="controlbutton" id="{divid}-timeshalf" style="margin-left:10px;display: inline-block;">
<button class="galpybutton">Speed<font face="Arial">&thinsp;</font>/<font face="Arial">&thinsp;</font>2</button></div>
<div class="controlbutton" id="{divid}-replay" style="margin-left:10px;display: inline-block;">
<button class="galpybutton">Replay</button></div>
{load_jslibs_code}

<script>
require.config({{
  paths: {{
    Plotly: 'https://cdn.plot.ly/plotly-latest.min',
  }}
}});
</script>

<script>
require(['Plotly'], function (Plotly) {{
{json_code}
  let layout = {layout};
  let numPerFrame= 5;    
  let cnt= 1;
  let interval;
  let trace_slice_len;
  let trace_slice_begin;
  let trace_slice_end;

  setup_trace();
  
  $('.controlbutton button').click(function() {{
    let button_type= this.parentNode.id;
    if ( button_type === '{divid}-play' ) {{
      clearInterval(interval);
      interval= animate_trace();
    }}
    else if ( button_type === '{divid}-pause' )
      clearInterval(interval);
    else if ( button_type === '{divid}-timestwo' ) {{
      cnt/= 2;
      numPerFrame*= 2;
    }}
    else if ( button_type === '{divid}-timeshalf' ) {{
      cnt*= 2;
      numPerFrame/= 2;
    }}
    else if ( button_type === '{divid}-replay' ) {{
      cnt= 1;
      try {{ // doesn't exist if animation has already ended
        Plotly.deleteTraces('{divid}',[{trace_num_20_list}]);
      }}
      catch (err) {{
      }}
        Plotly.deleteTraces('{divid}', {trace_num_list});
      clearInterval(interval);
      setup_trace();
      interval= animate_trace();
    }}
  }});
    
  function setup_trace() {{
    {setup_trace1}

    {setup_trace2}

    {setup_trace3}

    Plotly.plot('{divid}',traces,layout,{config});
  }}

  function animate_trace() {{
    return setInterval(function() {{
      // Make sure narrow and thick trace end in the same 
      // and the highlighted length has constant length
      trace_slice_len= Math.floor(numPerFrame);
      if ( trace_slice_len < 1) trace_slice_len= 1;
      trace_slice_begin= Math.floor(cnt*numPerFrame);
      trace_slice_end= Math.floor(Math.min(cnt*numPerFrame+trace_slice_len,data.x1_0.length-1));
      traces = {{x: [{x_data_list}], y: [{y_data_list}]}};
      Plotly.extendTraces('{divid}', traces, [{trace_num_10_list}]);
      trace_slice_begin-= trace_slice_len;
      traces = {{x: [{x_data_list}], y: [{y_data_list}]}};
      Plotly.restyle('{divid}', traces, [{trace_num_20_list}]);
      cnt+= 1;
      if(cnt*numPerFrame+trace_slice_len > data.x1_0.length/1) {{
          clearInterval(interval);
          Plotly.deleteTraces('{divid}',[{trace_num_20_list}]);
      }}
    }}, 30);
    }}
{close_json_code}}});
</script>""".format(json_code=json_code,close_json_code=close_json_code,
                    divid=self.divid,width=width,height=height,
                    button_margin_left=button_margin_left,config=config,
                    layout=layout,load_jslibs_code=load_jslibs_code,
                    x_data_list=x_data_list, y_data_list=y_data_list,
                    trace_num_10_list=trace_num_10_list, trace_num_20_list=trace_num_20_list,
                    setup_trace1=setup_trace1,setup_trace2=setup_trace2,
                    setup_trace3=setup_trace3, trace_num_list= [ii for ii in range(self.size * len(d1s))]))

class _1DInterp(object): 
    """Class to simulate 2D interpolation when using a single orbit"""
    def __init__(self,t,y,k=3):
        self._ip= interpolate.InterpolatedUnivariateSpline(t,y,k=k)
    def __call__(self,t,indx):
        return self._ip(t)[:,None]

def _from_name_oneobject(name,obs):
    """
    NAME:
       _from_name_oneobject
    PURPOSE:
       Query Simbad for the phase-space coordinates of one object
    INPUT:
       name - name of the object
       obs - numpy.array of [ro,vo,zo,solarmotion] that can be altered
    OUTPUT:
       [ra,dec,dist,pmra,pmdec,vlos]
    HISTORY:
       2018-07-15 - Written - Mathew Bub (UofT)
       2019-06-16 - Added named_objects - Bovy (UofT)
    """
    # First check whether this is a named_object
    this_name= _named_objects_key_formatting(name)
    # Find the object in the file?
    if this_name in _known_objects.keys():
        if 'ra' in _known_objects[this_name].keys():
            vxvv= [_known_objects[this_name]['ra'],
                   _known_objects[this_name]['dec'],
                   _known_objects[this_name]['distance'],
                   _known_objects[this_name]['pmra'],
                   _known_objects[this_name]['pmdec'],
                   _known_objects[this_name]['vlos']]
        # If you add another way, need to convert to ra,dec,... bc from_name
        # expects that
        if obs[0] is None and \
                'ro' in _known_objects[this_name].keys():
            obs[0]= _known_objects[this_name]['ro']
        if obs[1] is None and \
                'vo' in _known_objects[this_name].keys():
            obs[1]= _known_objects[this_name]['vo']
        if obs[2] is None and \
                'zo' in _known_objects[this_name].keys():
            obs[2]= _known_objects[this_name]['zo']
        if obs[3] is None and \
                'solarmotion' in _known_objects[this_name].keys():
            obs[3]= _known_objects[this_name]['solarmotion']
        return vxvv
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
        dist= 1./simbad_table['PLX_VALUE'][0]
    else:
        dist_str= str(simbad_table['Distance_distance'][0]) + \
            simbad_table['Distance_unit'][0]
        dist= units.Quantity(dist_str).to(units.kpc).value
    return [ra,dec,dist,pmra,pmdec,vlos]

def _fit_orbit(orb,vxvv,vxvv_err,pot,radec=False,lb=False,
               customsky=False,lb_to_customsky=None,
               pmllpmbb_to_customsky=None,
               tintJ=100,ntintJ=1000,integrate_method='dopr54_c',
               ro=None,vo=None,obs=None,disp=False):
    """Fit an orbit to data in a given potential"""
    # Need to turn this off for speed
    coords._APY_COORDS_ORIG= coords._APY_COORDS
    coords._APY_COORDS= False
    #Import here, because otherwise there is an infinite loop of imports
    from ..actionAngle import actionAngleIsochroneApprox, actionAngle
    #Mock this up, bc we want to use its orbit-integration routines
    class mockActionAngleIsochroneApprox(actionAngleIsochroneApprox):
        def __init__(self,tintJ,ntintJ,pot,integrate_method='dopr54_c'):
            actionAngle.__init__(self)
            self._tintJ= tintJ
            self._ntintJ=ntintJ
            self._tsJ= numpy.linspace(0.,self._tintJ,self._ntintJ)
            self._integrate_dt= None
            self._pot= pot
            self._integrate_method= integrate_method
            return None
    tmockAA= mockActionAngleIsochroneApprox(tintJ,ntintJ,pot,
                                            integrate_method=integrate_method)
    opt_vxvv= optimize.fmin_powell(_fit_orbit_mlogl,orb.vxvv,
                                   args=(vxvv,vxvv_err,pot,radec,lb,
                                         customsky,lb_to_customsky,
                                         pmllpmbb_to_customsky,
                                         tmockAA,
                                         ro,vo,obs),
                                   disp=disp)
    maxLogL= -_fit_orbit_mlogl(opt_vxvv,vxvv,vxvv_err,pot,radec,lb,
                               customsky,lb_to_customsky,pmllpmbb_to_customsky,
                               tmockAA,
                               ro,vo,obs)
    coords._APY_COORDS= coords._APY_COORDS_ORIG
    return (opt_vxvv,maxLogL)

def _fit_orbit_mlogl(new_vxvv,vxvv,vxvv_err,pot,radec,lb,
                     customsky,lb_to_customsky,pmllpmbb_to_customsky,
                     tmockAA,
                     ro,vo,obs):
    """The log likelihood for fitting an orbit"""
    #Use this _parse_args routine, which does forward and backward integration
    iR,ivR,ivT,iz,ivz,iphi= tmockAA._parse_args(True,False,
                                                new_vxvv[0],
                                                new_vxvv[1],
                                                new_vxvv[2],
                                                new_vxvv[3],
                                                new_vxvv[4],
                                                new_vxvv[5])
    if radec or lb or customsky:
        #Need to transform to (l,b), (ra,dec), or a custom set
        #First transform to X,Y,Z,vX,vY,vZ (Galactic)
        X,Y,Z = coords.galcencyl_to_XYZ(iR.flatten(),iphi.flatten(),
                                        iz.flatten(),
                                        Xsun=obs[0]/ro,
                                        Zsun=obs[2]/ro).T
        vX,vY,vZ = coords.galcencyl_to_vxvyvz(ivR.flatten(),ivT.flatten(),
                                              ivz.flatten(),iphi.flatten(),
                                              vsun=numpy.array(\
                obs[3:6])/vo,Xsun=obs[0]/ro,Zsun=obs[2]/ro).T
        bad_indx= (X == 0.)*(Y == 0.)*(Z == 0.)
        if True in bad_indx: X[bad_indx]+= ro/10000.
        lbdvrpmllpmbb= coords.rectgal_to_sphergal(X*ro,Y*ro,Z*ro,
                                                  vX*vo,vY*vo,vZ*vo,
                                                  degree=True)
        if lb:
            orb_vxvv= numpy.array([lbdvrpmllpmbb[:,0],
                                   lbdvrpmllpmbb[:,1],
                                   lbdvrpmllpmbb[:,2],
                                   lbdvrpmllpmbb[:,4],
                                   lbdvrpmllpmbb[:,5],
                                   lbdvrpmllpmbb[:,3]]).T
        elif radec:
            #Further transform to ra,dec,pmra,pmdec
            radec= coords.lb_to_radec(lbdvrpmllpmbb[:,0],
                                      lbdvrpmllpmbb[:,1],degree=True,
                                      epoch=None)
            pmrapmdec= coords.pmllpmbb_to_pmrapmdec(lbdvrpmllpmbb[:,4],
                                                    lbdvrpmllpmbb[:,5],
                                                    lbdvrpmllpmbb[:,0],
                                                    lbdvrpmllpmbb[:,1],
                                                    degree=True,
                                                    epoch=None)
            orb_vxvv= numpy.array([radec[:,0],radec[:,1],
                                   lbdvrpmllpmbb[:,2],
                                   pmrapmdec[:,0],pmrapmdec[:,1],
                                   lbdvrpmllpmbb[:,3]]).T
        elif customsky:
            #Further transform to ra,dec,pmra,pmdec
            customradec= lb_to_customsky(lbdvrpmllpmbb[:,0],
                                              lbdvrpmllpmbb[:,1],degree=True)
            custompmrapmdec= pmllpmbb_to_customsky(lbdvrpmllpmbb[:,4],
                                                   lbdvrpmllpmbb[:,5],
                                                   lbdvrpmllpmbb[:,0],
                                                   lbdvrpmllpmbb[:,1],
                                                   degree=True)
            orb_vxvv= numpy.array([customradec[:,0],customradec[:,1],
                                   lbdvrpmllpmbb[:,2],
                                   custompmrapmdec[:,0],custompmrapmdec[:,1],
                                   lbdvrpmllpmbb[:,3]]).T
    else:
        #shape=(2tintJ-1,6)
        orb_vxvv= numpy.array([iR.flatten(),ivR.flatten(),ivT.flatten(),
                            iz.flatten(),ivz.flatten(),iphi.flatten()]).T 
    out= 0.
    for ii in range(vxvv.shape[0]):
        sub_vxvv= (orb_vxvv-vxvv[ii,:].flatten())**2.
        #print(sub_vxvv[numpy.argmin(numpy.sum(sub_vxvv,axis=1))])
        if not vxvv_err is None:
            sub_vxvv/= vxvv_err[ii,:]**2.
        else:
            sub_vxvv/= 0.01**2.
        out+= logsumexp(-0.5*numpy.sum(sub_vxvv,axis=1))
    return -out

def _check_roSet(orb,kwargs,funcName):
    """Function to check whether ro is set, because it's required for funcName"""
    if not orb._roSet and kwargs.get('ro',None) is None:
        warnings.warn("Method %s(.) requires ro to be given at Orbit initialization or at method evaluation; using default ro which is %f kpc" % (funcName,orb._ro),
                      galpyWarning)

def _check_voSet(orb,kwargs,funcName):
    """Function to check whether vo is set, because it's required for funcName"""
    if not orb._voSet and kwargs.get('vo',None) is None:
        warnings.warn("Method %s(.) requires vo to be given at Orbit initialization or at method evaluation; using default vo which is %f km/s" % (funcName,orb._vo),
                      galpyWarning)

def _helioXYZ(orb,thiso,*args,**kwargs):
    """Calculate heliocentric rectangular coordinates"""
    obs, ro, vo= _parse_radec_kwargs(orb,kwargs)
    if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
    if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6: #pragma: no cover
        raise AttributeError("orbit must track azimuth to use radeclbd functions")
    elif len(thiso[:,0]) == 4: #planarOrbit
        if isinstance(obs,(numpy.ndarray,list)):
            X,Y,Z = coords.galcencyl_to_XYZ(\
                thiso[0,:],thiso[3,:]-numpy.arctan2(obs[1],obs[0]),0.,
                Xsun=numpy.sqrt(obs[0]**2.+obs[1]**2.)/ro,
                Zsun=obs[2]/ro,_extra_rot=False).T
        else: #Orbit instance
            obs.turn_physical_off()
            if obs.dim() == 2:
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[3,:]-obs.phi(*args,**kwargs),
                    numpy.zeros_like(thiso[0]),
                    Xsun=obs.R(*args,**kwargs),Zsun=0.,_extra_rot=False).T
            else:
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[3,:]-obs.phi(*args,**kwargs),
                    numpy.zeros_like(thiso[0]),
                    Xsun=obs.R(*args,**kwargs),
                    Zsun=obs.z(*args,**kwargs),_extra_rot=False).T
            obs.turn_physical_on()
    else: #FullOrbit
        if isinstance(obs,(numpy.ndarray,list)):
            X,Y,Z = coords.galcencyl_to_XYZ(\
                thiso[0,:],thiso[5,:]-numpy.arctan2(obs[1],obs[0]),
                thiso[3,:],
                Xsun=numpy.sqrt(obs[0]**2.+obs[1]**2.)/ro,
                Zsun=obs[2]/ro).T
        else: #Orbit instance
            obs.turn_physical_off()
            if obs.dim() == 2:
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[5,:]-obs.phi(*args,**kwargs),
                    thiso[3,:],
                    Xsun=obs.R(*args,**kwargs),Zsun=0.).T
            else:
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[5,:]-obs.phi(*args,**kwargs),
                    thiso[3,:],
                    Xsun=obs.R(*args,**kwargs),
                    Zsun=obs.z(*args,**kwargs)).T
            obs.turn_physical_on()
    return (X*ro,Y*ro,Z*ro)

def _lbd(orb,thiso,*args,**kwargs):
    """Calculate l,b, and d"""
    obs, ro, vo= _parse_radec_kwargs(orb,kwargs,dontpop=True)
    X,Y,Z= _helioXYZ(orb,thiso,*args,**kwargs)
    bad_indx= (X == 0.)*(Y == 0.)*(Z == 0.)
    if True in bad_indx:
        X[bad_indx]+= 1e-15
    return coords.XYZ_to_lbd(X,Y,Z,degree=True)

def _radec(orb,thiso,*args,**kwargs):
    """Calculate ra and dec"""
    lbd= _lbd(orb,thiso,*args,**kwargs)
    return coords.lb_to_radec(lbd[:,0],lbd[:,1],degree=True,epoch=None)

def _XYZvxvyvz(orb,thiso,*args,**kwargs):
    """Calculate X,Y,Z,U,V,W"""
    obs, ro, vo= _parse_radec_kwargs(orb,kwargs,vel=True)
    if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
    if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6: #pragma: no cover
        raise AttributeError("orbit must track azimuth to use radeclbduvw functions")
    elif len(thiso[:,0]) == 4: #planarOrbit
        if isinstance(obs,(numpy.ndarray,list)):
            Xsun= numpy.sqrt(obs[0]**2.+obs[1]**2.)
            X,Y,Z = coords.galcencyl_to_XYZ(\
                thiso[0,:],thiso[3,:]-numpy.arctan2(obs[1],obs[0]),
                numpy.zeros_like(thiso[0]),
                Xsun=Xsun/ro,Zsun=obs[2]/ro,_extra_rot=False).T
            vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                thiso[1,:],thiso[2,:],numpy.zeros_like(thiso[0]),
                thiso[3,:]-numpy.arctan2(obs[1],obs[0]),
                vsun=numpy.array(# have to rotate
                    [obs[3]*obs[0]/Xsun+obs[4]*obs[1]/Xsun,
                     -obs[3]*obs[1]/Xsun+obs[4]*obs[0]/Xsun,
                     obs[5]])/vo,
                Xsun=Xsun/ro,Zsun=obs[2]/ro,_extra_rot=False).T
        else: #Orbit instance
            obs.turn_physical_off()
            if obs.dim() == 2:
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[3,:]-obs.phi(*args,**kwargs),
                    numpy.zeros_like(thiso[0]),
                    Xsun=obs.R(*args,**kwargs),Zsun=0.,_extra_rot=False).T
                vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                    thiso[1,:],thiso[2,:],numpy.zeros_like(thiso[0]),
                    thiso[3,:]-obs.phi(*args,**kwargs),
                    vsun=numpy.array([\
                            obs.vR(*args,**kwargs),obs.vT(*args,**kwargs),
                            0.]),
                    Xsun=obs.R(*args,**kwargs),Zsun=0.,_extra_rot=False).T
            else:
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[3,:]-obs.phi(*args,**kwargs),
                    numpy.zeros_like(thiso[0]),
                    Xsun=obs.R(*args,**kwargs),
                    Zsun=obs.z(*args,**kwargs),_extra_rot=False).T
                vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                    thiso[1,:],thiso[2,:],numpy.zeros_like(thiso[0]),
                    thiso[3,:]-obs.phi(*args,**kwargs),
                    vsun=numpy.array([\
                            obs.vR(*args,**kwargs),
                            obs.vT(*args,**kwargs),
                            obs.vz(*args,**kwargs)]),
                    Xsun=obs.R(*args,**kwargs),
                    Zsun=obs.z(*args,**kwargs),_extra_rot=False).T
            obs.turn_physical_on()
    else: #FullOrbit
        if isinstance(obs,(numpy.ndarray,list)):
            Xsun= numpy.sqrt(obs[0]**2.+obs[1]**2.)
            X,Y,Z = coords.galcencyl_to_XYZ(\
                thiso[0,:],thiso[5,:]-numpy.arctan2(obs[1],obs[0]),thiso[3,:],
                Xsun=Xsun/ro,Zsun=obs[2]/ro).T
            vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                thiso[1,:],thiso[2,:],thiso[4,:],
                thiso[5,:]-numpy.arctan2(obs[1],obs[0]),
                vsun=numpy.array(# have to rotate
                    [obs[3]*obs[0]/Xsun+obs[4]*obs[1]/Xsun,
                     -obs[3]*obs[1]/Xsun+obs[4]*obs[0]/Xsun,
                     obs[5]])/vo,
                Xsun=Xsun/ro,Zsun=obs[2]/ro).T
        else: #Orbit instance
            obs.turn_physical_off()
            if obs.dim() == 2:
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[5,:]-obs.phi(*args,**kwargs),
                    thiso[3,:],
                    Xsun=obs.R(*args,**kwargs),Zsun=0.).T
                vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                    thiso[1,:],thiso[2,:],thiso[4,:],
                    thiso[5,:]-obs.phi(*args,**kwargs),
                    vsun=numpy.array([\
                            obs.vR(*args,**kwargs),obs.vT(*args,**kwargs),
                            0.]),
                    Xsun=obs.R(*args,**kwargs),Zsun=0.).T
            else:
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[5,:]-obs.phi(*args,**kwargs),
                    thiso[3,:],
                    Xsun=obs.R(*args,**kwargs),
                    Zsun=obs.z(*args,**kwargs)).T
                vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                    thiso[1,:],thiso[2,:],thiso[4,:],
                    thiso[5,:]-obs.phi(*args,**kwargs),
                    vsun=numpy.array([\
                            obs.vR(*args,**kwargs),
                            obs.vT(*args,**kwargs),
                            obs.vz(*args,**kwargs)]),
                    Xsun=obs.R(*args,**kwargs),
                    Zsun=obs.z(*args,**kwargs)).T
            obs.turn_physical_on()
    return (X*ro,Y*ro,Z*ro,vX*vo,vY*vo,vZ*vo)

def _lbdvrpmllpmbb(orb,thiso,*args,**kwargs):
    """Calculate l,b,d,vr,pmll,pmbb"""
    obs, ro, vo= _parse_radec_kwargs(orb,kwargs,dontpop=True)
    X,Y,Z,vX,vY,vZ= _XYZvxvyvz(orb,thiso,*args,**kwargs)
    bad_indx= (X == 0.)*(Y == 0.)*(Z == 0.)
    if True in bad_indx:
        X[bad_indx]+= ro/10000.
    return coords.rectgal_to_sphergal(X,Y,Z,vX,vY,vZ,degree=True)

def _pmrapmdec(orb,thiso,*args,**kwargs):
    """Calculate pmra and pmdec"""
    lbdvrpmllpmbb= _lbdvrpmllpmbb(orb,thiso,*args,**kwargs)
    return coords.pmllpmbb_to_pmrapmdec(lbdvrpmllpmbb[:,4],
                                        lbdvrpmllpmbb[:,5],
                                        lbdvrpmllpmbb[:,0],
                                        lbdvrpmllpmbb[:,1],degree=True,
                                        epoch=None)

def _parse_radec_kwargs(orb,kwargs,vel=False,dontpop=False):
    if 'obs' in kwargs:
        obs= kwargs['obs']
        if not dontpop:
            kwargs.pop('obs')
        if isinstance(obs,(list,numpy.ndarray)):
            if len(obs) == 2:
                obs= [obs[0],obs[1],0.]
            elif len(obs) == 4:
                obs= [obs[0],obs[1],0.,obs[2],obs[3],0.]
            for ii in range(len(obs)):
                if _APY_LOADED and isinstance(obs[ii],units.Quantity):
                    if ii < 3:
                        obs[ii]= conversion.parse_length_kpc(obs[ii])
                    else:
                        obs[ii]= conversion.parse_velocity_kms(obs[ii])
    else:
        if vel:
            obs= [orb._ro,0.,orb._zo,
                  orb._solarmotion[0],orb._solarmotion[1]+orb._vo,
                  orb._solarmotion[2]]
        else:
            obs= [orb._ro,0.,orb._zo]
    if 'ro' in kwargs:
        ro= conversion.parse_length_kpc(kwargs['ro'])
        if not dontpop:
            kwargs.pop('ro')
    else:
        ro= orb._ro
    if 'vo' in kwargs:
        vo= conversion.parse_velocity_kms(kwargs['vo'])
        if not dontpop:
            kwargs.pop('vo')
    else:
        vo= orb._vo
    return (obs,ro,vo)

def _check_integrate_dt(t,dt):
    """Check that the stepsize in t is an integer x dt"""
    if dt is None:
        return True
    mult= round((t[1]-t[0])/dt)
    if numpy.fabs(mult*dt-t[1]+t[0]) < 10.**-10.:
        return True
    else:
        return False

def _check_potential_dim(orb,pot):
    from ..potential import _dim
    # Don't deal with pot=None here, just dimensionality
    assert pot is None or orb.dim() <= _dim(pot), 'Orbit dimensionality is %i, but potential dimensionality is %i < %i; orbit needs to be of equal or lower dimensionality as the potential; you can reduce the dimensionality---if appropriate---of your orbit with orbit.toPlanar or orbit.toVertical' % (orb.dim(),_dim(pot),orb.dim())
    assert pot is None or not (orb.dim() == 1 and _dim(pot) != 1), 'Orbit dimensionality is 1, but potential dimensionality is %i != 1; 1D orbits can only be integrated in 1D potentials; you convert your potential to a 1D potential---if appropriate---using potential.toVerticalPotential' % (_dim(pot))

def _check_consistent_units(orb,pot):
    if pot is None: return None
    assert physical_compatible(orb,pot), 'Physical conversion for the Orbit object is not consistent with that of the Potential given to it'
